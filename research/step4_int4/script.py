"""
STEP 4 -- INT4 Weight Quantisation (W4A16, pure PyTorch)
=========================================================
Research: Memory Optimisation for Generative AI
Hardware: NVIDIA RTX 3090 (24 GB VRAM)
Resolution: 768 x 1024
Strategy: Compress weights from int8 (1 byte) to int4 (0.5 bytes) via bit packing.
          Combined with xformers from Step 1.

What changes from Step 2:
  QuantizedLinearInt4 replaces QuantizedLinear (int8)
  Bit packing: two 4-bit values stored per byte
  Scale range: max / 7.0  (int4 symmetric range is -7 to 7)

What stays the same:
  xformers attention, same garment, same model, same resolution.

Trade-off vs INT8:
  - Theoretical extra saving: ~3 GB weight memory
  - Risk: higher rounding error per weight (14 levels vs 254 levels)
  - Quality may visibly degrade -- this run documents whether it is acceptable

Expected result vs Step 2 (11.29 GB):
  - VRAM    : ~8-9 GB (another ~3 GB from halving bits per weight)
  - Speed   : similar
  - Quality : unknown -- the point of this experiment
"""

import sys, os, time, json, gc
from pathlib import Path

ROOT     = str(Path(__file__).parent.parent.parent.resolve())
IDMVTON  = os.path.join(ROOT, 'IDM-VTON')
DEMO_DIR = os.path.join(IDMVTON, 'gradio_demo')
sys.path.insert(0, IDMVTON)
sys.path.insert(0, DEMO_DIR)

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from transformers import (
    AutoTokenizer, CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel, CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from utils_mask import get_mask_location
import apply_net
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

# ── Configuration ───────────────────────────────────────────────────────────────
MODEL_PATH    = 'yisol/IDM-VTON'
GARMENT_IMG   = os.path.join(ROOT, 'research', 'denimshirt1.png')
HUMAN_IMG     = os.path.join(DEMO_DIR, 'example', 'human', '00034_00.jpg')
OUTPUT_DIR    = os.path.join(Path(__file__).parent.resolve(), 'results')
GARMENT_DESC  = 'blue denim shirt'
WIDTH, HEIGHT = 768, 1024
STEPS         = 30
GUIDANCE      = 2.0
SEED          = 42
DTYPE         = torch.float16
DEVICE        = 'cuda:0'

DENSEPOSE_CFG = os.path.join(IDMVTON, 'configs', 'densepose_rcnn_R_50_FPN_s1x.yaml')
DENSEPOSE_PKL = os.path.join(IDMVTON, 'ckpt', 'densepose', 'model_final_162be9.pkl')

BASELINE_VRAM = 16.29
STEP2_VRAM    = 11.29

os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg): print(f'[STEP4] {msg}', flush=True)
def vram_gb(): return torch.cuda.max_memory_allocated() / 1024**3

tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def prepare_garment(img_path, width, height):
    img = Image.open(img_path)
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    else:
        img = img.convert('RGB')
    scale = min(width / img.width, height / img.height)
    new_w, new_h = int(img.width * scale), int(img.height * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new('RGB', (width, height), (255, 255, 255))
    canvas.paste(img, ((width - new_w) // 2, (height - new_h) // 2))
    return canvas


# ── INT4 Quantisation Implementation ───────────────────────────────────────────
#
# INT4 stores each weight in 4 bits instead of 8.
# Range: -7 to +7 (symmetric, 15 levels vs 255 for INT8).
# Two values are bit-packed into one byte: low nibble + high nibble.
#
# Memory: INT8 uses 1 byte/weight. INT4 uses 0.5 bytes/weight.
# Error:  rounding error per weight is ~2x larger than INT8.
#
# Packing scheme:
#   value in [-7, 7]  ->  shift to [0, 14]  ->  store in 4 bits
#   byte = low_nibble | (high_nibble << 4)
#
class QuantizedLinearInt4(nn.Module):
    def __init__(self, in_features, out_features, weight_fp16, bias=None):
        super().__init__()
        self.in_features = in_features
        w = weight_fp16.float()

        # Per-channel symmetric scale: map max abs weight to 7
        scale = w.abs().max(dim=1, keepdim=True).values / 7.0
        scale = scale.clamp(min=1e-8)

        # Quantise to [-7, 7]
        w_q = (w / scale).round().clamp(-7, 7).to(torch.int8)

        # Shift to unsigned [0, 14] for nibble packing
        w_u = w_q.to(torch.int16) + 7

        # Pad to even number of columns so we can pair them cleanly
        if in_features % 2 != 0:
            pad = torch.zeros(out_features, 1, dtype=torch.int16)
            w_u = torch.cat([w_u, pad], dim=1)

        # Pack: even-column value -> low nibble, odd-column value -> high nibble
        low  = w_u[:, 0::2]           # [out, ceil(in/2)]
        high = w_u[:, 1::2]           # [out, ceil(in/2)]
        packed = (low | (high << 4)).to(torch.uint8).view(torch.int8)

        self.register_buffer('weight_packed', packed)
        self.register_buffer('scale', scale.squeeze(1).half())
        if bias is not None:
            self.register_buffer('bias', bias.half())
        else:
            self.bias = None

    def forward(self, x, *args, **kwargs):
        # Reinterpret stored int8 as unsigned to avoid sign-extension issues
        packed = self.weight_packed.to(torch.int16) & 0xFF

        # Extract nibbles
        low  = (packed & 0xF)          # even columns [0, 14]
        high = (packed >> 4) & 0xF     # odd columns  [0, 14]

        # Interleave back into full weight matrix
        out_f, n_packed = low.shape
        w_u = torch.empty(out_f, n_packed * 2, dtype=torch.int16, device=x.device)
        w_u[:, 0::2] = low
        w_u[:, 1::2] = high

        # Trim padding and unshift back to [-7, 7]
        w_q = w_u[:, :self.in_features] - 7

        # Dequantise to fp16 and run matmul
        w_fp = w_q.to(x.dtype) * self.scale.unsqueeze(1)
        return nn.functional.linear(x, w_fp, self.bias)


def quantize_to_int4(model, min_params=2048):
    """Replace every nn.Linear >= min_params with QuantizedLinearInt4."""
    replaced = 0
    before_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if module.weight.numel() < min_params:
            continue

        parts = name.rsplit('.', 1)
        parent = model
        for part in parts[:-1]:
            for seg in part.split('.'):
                parent = getattr(parent, seg)
        child_name = parts[-1] if len(parts) > 1 else name

        new_layer = QuantizedLinearInt4(
            module.in_features,
            module.out_features,
            module.weight.data,
            module.bias.data if module.bias is not None else None,
        )
        setattr(parent, child_name, new_layer)
        replaced += 1

    after_mb = sum(
        (p.numel() * p.element_size()) for p in model.parameters()
    ) / 1024**2
    # Buffers hold the packed weights -- count those too
    after_mb += sum(
        (b.numel() * b.element_size()) for b in model.buffers()
    ) / 1024**2

    return replaced, round(before_mb), round(after_mb)


# ── Prerequisite Check ──────────────────────────────────────────────────────────
if not os.path.exists(DENSEPOSE_PKL):
    raise FileNotFoundError(f"DensePose checkpoint missing:\n  {DENSEPOSE_PKL}")
for onnx in ('parsing_atr.onnx', 'parsing_lip.onnx'):
    p = os.path.join(IDMVTON, 'ckpt', 'humanparsing', onnx)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Human parsing model missing:\n  {p}")

# ── Stage 1: Preprocessing Models ──────────────────────────────────────────────
log('Loading OpenPose...')
openpose_model = OpenPose(0)
openpose_model.preprocessor.body_estimation.model.to(DEVICE)

log('Loading Human Parsing (ONNX)...')
parsing_model = Parsing(0)

# ── Stage 2: Load to CPU, Quantise, Move to GPU ─────────────────────────────────
log('Loading IDM-VTON models to CPU (for INT4 quantisation)...')
torch.cuda.reset_peak_memory_stats()
t_load = time.time()

tokenizer_one   = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder='tokenizer',   use_fast=False)
tokenizer_two   = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder='tokenizer_2', use_fast=False)
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder='scheduler')

text_enc1 = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder='text_encoder',   torch_dtype=DTYPE)
text_enc2 = CLIPTextModelWithProjection.from_pretrained(MODEL_PATH, subfolder='text_encoder_2', torch_dtype=DTYPE)
img_enc   = CLIPVisionModelWithProjection.from_pretrained(MODEL_PATH, subfolder='image_encoder', torch_dtype=DTYPE)
vae       = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder='vae', torch_dtype=DTYPE)

log('Loading UNet encoder (garment)...')
unet_enc = UNet2DConditionModel_ref.from_pretrained(MODEL_PATH, subfolder='unet_encoder', torch_dtype=DTYPE)
unet_enc.requires_grad_(False)

log('Loading UNet (main diffusion)...')
unet = UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder='unet', torch_dtype=DTYPE)
unet.requires_grad_(False)

# ── INT4 Quantisation ───────────────────────────────────────────────────────────
log('Quantising UNet linear layers to INT4...')
n, before, after = quantize_to_int4(unet)
log(f'  UNet: {n} layers quantised | {before} MB -> {after} MB (saved {before - after} MB)')

log('Quantising garment UNet encoder linear layers to INT4...')
n2, before2, after2 = quantize_to_int4(unet_enc)
log(f'  UNet encoder: {n2} layers quantised | {before2} MB -> {after2} MB (saved {before2 - after2} MB)')

total_saved = (before + before2) - (after + after2)
log(f'Total weight memory saved: {total_saved} MB ({total_saved/1024:.2f} GB)')

# ── Assemble Pipeline and Move to GPU ──────────────────────────────────────────
log('Assembling pipeline and moving to GPU...')
pipe = TryonPipeline.from_pretrained(
    MODEL_PATH,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_enc1,
    text_encoder_2=text_enc2,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=img_enc,
    torch_dtype=DTYPE,
)
pipe.unet_encoder = unet_enc
pipe.to(DEVICE)
pipe.unet_encoder.to(DEVICE)

# xformers from Step 1
log('Enabling xformers memory-efficient attention...')
try:
    pipe.enable_xformers_memory_efficient_attention()
    pipe.unet_encoder.enable_xformers_memory_efficient_attention()
    log('  xformers: ENABLED on both UNets')
except Exception as e:
    log(f'  xformers WARNING: {e}')

log('VAE tiling: disabled')

load_time = time.time() - t_load
log(f'Ready in {load_time:.1f}s  |  VRAM after load: {vram_gb():.2f} GB')

# ── Stage 3: Preprocess Inputs ──────────────────────────────────────────────────
log('Preprocessing inputs...')
garm_img  = prepare_garment(GARMENT_IMG, WIDTH, HEIGHT)
human_img = Image.open(HUMAN_IMG).convert('RGB').resize((WIDTH, HEIGHT))

keypoints      = openpose_model(human_img.resize((384, 512)))
model_parse, _ = parsing_model(human_img.resize((384, 512)))
mask, mask_gray = get_mask_location('hd', 'upper_body', model_parse, keypoints)
mask = mask.resize((WIDTH, HEIGHT))
mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

human_arg = _apply_exif_orientation(human_img.resize((384, 512)))
human_arg = convert_PIL_to_numpy(human_arg, format='BGR')

log('Running DensePose...')
dp_args = apply_net.create_argument_parser().parse_args((
    'show', DENSEPOSE_CFG, DENSEPOSE_PKL,
    'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda',
))
pose_img = dp_args.func(dp_args, human_arg)[:, :, ::-1]
pose_img = Image.fromarray(pose_img).resize((WIDTH, HEIGHT))

# ── Stage 4: Inference ──────────────────────────────────────────────────────────
log(f'Running inference ({STEPS} steps) with INT4 + xformers...')
torch.cuda.reset_peak_memory_stats()
t_inf = time.time()

with torch.no_grad():
    with torch.cuda.amp.autocast():
        prompt          = f'model is wearing {GARMENT_DESC}'
        negative_prompt = 'monochrome, lowres, bad anatomy, worst quality, low quality'

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

        prompt_c = f'a photo of {GARMENT_DESC}'
        with torch.inference_mode():
            (prompt_embeds_c, _, _, _) = pipe.encode_prompt(
                prompt_c,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=negative_prompt,
            )

        pose_img_t  = tensor_transform(pose_img).unsqueeze(0).to(DEVICE, DTYPE)
        garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(DEVICE, DTYPE)
        generator   = torch.Generator(DEVICE).manual_seed(SEED)

        images = pipe(
            prompt_embeds=prompt_embeds.to(DEVICE, DTYPE),
            negative_prompt_embeds=negative_prompt_embeds.to(DEVICE, DTYPE),
            pooled_prompt_embeds=pooled_prompt_embeds.to(DEVICE, DTYPE),
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(DEVICE, DTYPE),
            num_inference_steps=STEPS,
            generator=generator,
            strength=1.0,
            pose_img=pose_img_t,
            text_embeds_cloth=prompt_embeds_c.to(DEVICE, DTYPE),
            cloth=garm_tensor,
            mask_image=mask,
            image=human_img,
            height=HEIGHT,
            width=WIDTH,
            ip_adapter_image=garm_img.resize((WIDTH, HEIGHT)),
            guidance_scale=GUIDANCE,
        )[0]

inf_time  = time.time() - t_inf
vram_peak = vram_gb()

# ── Stage 5: Save Outputs ───────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, 'tryon_output.png')
images[0].save(out_path)
mask_gray.save(os.path.join(OUTPUT_DIR, 'mask_preview.png'))
human_img.save(os.path.join(OUTPUT_DIR, 'human_input.png'))
garm_img.save(os.path.join(OUTPUT_DIR, 'garment_input.png'))

metrics = {
    'step':                   'step4_int4',
    'gpu':                    'RTX 3090 24GB',
    'optimisations':          ['int4_weights', 'xformers_memory_efficient_attention'],
    'dtype':                  'float16 activations, int4 weights',
    'resolution':             f'{WIDTH}x{HEIGHT}',
    'steps':                  STEPS,
    'guidance_scale':         GUIDANCE,
    'seed':                   SEED,
    'vram_peak_gb':           round(vram_peak, 2),
    'vram_saved_vs_baseline_gb': round(BASELINE_VRAM - vram_peak, 2),
    'vram_saved_vs_step2_gb': round(STEP2_VRAM - vram_peak, 2),
    'inference_time_s':       round(inf_time, 1),
    'unet_layers_quantised':  n,
    'unet_enc_layers_quantised': n2,
    'weight_memory_before_mb': before + before2,
    'weight_memory_after_mb':  after + after2,
    'weight_memory_saved_mb':  total_saved,
    'baseline_vram_gb':       BASELINE_VRAM,
    'step2_vram_gb':          STEP2_VRAM,
}
with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

log('')
log('=' * 65)
log(f'  Step          : 4 -- INT4 Quantisation + xformers')
log(f'  Resolution    : {WIDTH}x{HEIGHT}  |  Steps: {STEPS}')
log(f'  VRAM peak     : {vram_peak:.2f} GB  (baseline: {BASELINE_VRAM} GB)')
log(f'  Saved vs base : {BASELINE_VRAM - vram_peak:.2f} GB')
log(f'  Saved vs step2: {STEP2_VRAM - vram_peak:.2f} GB')
log(f'  Weight memory : {total_saved} MB saved by INT4 packing')
log(f'  Infer time    : {inf_time:.1f}s')
log(f'  Output        : {out_path}')
log('=' * 65)
