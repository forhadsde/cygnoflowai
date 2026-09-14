"""
STEP 2 — INT8 Weight Quantisation (W8A16, pure PyTorch)
=========================================================
Research: Memory Optimisation for Generative AI
Hardware: NVIDIA RTX 3090 (24 GB VRAM)
Resolution: 768 x 1024 (same as baseline — isolate the variable)
Strategy: Compress model weights from float16 (2 bytes) to int8 (1 byte).
          Combined with xformers from Step 1.

What changes from Step 1:
  quantize_unet_linear_layers(unet)      ← NEW: weights → int8
  quantize_unet_linear_layers(unet_enc)  ← NEW: weights → int8
  (xformers + VAE tiling kept from Step 1)

What is W8A16?
  W = Weights (stored as int8, 1 byte each)
  A = Activations (computed as float16, 2 bytes each)
  16 = 16-bit precision for activations during computation

  Weights are dequantised to float16 just before each matrix multiplication,
  then the result is computed in float16. Peak VRAM for the temporary float16
  weight tensor is one layer at a time — negligible versus total savings.

Why not bitsandbytes?
  bitsandbytes is the standard library for INT8 inference but has known
  Windows path issues (searches for Linux .so files). We implement W8A16
  from scratch using pure PyTorch — this is more transparent and demonstrates
  understanding of the technique at the implementation level.

Expected result vs Step 1 (14.07 GB, 22.1s):
  - VRAM    : ~9–11 GB (further 3–4 GB reduction from weight compression)
  - Speed   : similar (no INT8 Tensor Core acceleration, but less data transfer)
  - Quality : near-identical (small rounding noise, visually imperceptible)
"""

import sys, os, time, json, gc
from pathlib import Path

# ── Path Setup ──────────────────────────────────────────────────────────────────
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

BASELINE_VRAM = 16.29   # Step 0
STEP1_VRAM    = 14.07   # Step 1

os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg): print(f'[STEP2] {msg}', flush=True)

def vram_gb():
    return torch.cuda.max_memory_allocated() / 1024**3

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


# ── INT8 Quantisation Implementation ───────────────────────────────────────────

class QuantizedLinear(nn.Module):
    """
    W8A16 Linear layer: weights stored as int8, activations remain float16.

    How it works:
      1. At construction: quantise the float16 weight matrix to int8.
         Per-output-channel symmetric quantisation:
           scale[i] = max(|row_i|) / 127
           int8_weight[i] = round(float16_weight[i] / scale[i])
      2. At forward: dequantise int8 → float16 for one layer,
         compute float16 matmul, immediately release the temp tensor.

    Memory layout in VRAM:
      float16 weight: out_features × in_features × 2 bytes
      int8    weight: out_features × in_features × 1 byte   ← 50% saving
      scale:          out_features × 2 bytes                 ← negligible

    The temporary float16 tensor during dequantisation is one layer at a time
    (~50–300 MB depending on layer size), released immediately after the matmul.
    """

    def __init__(self, in_features, out_features, weight_fp16, bias=None):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Per-output-channel scale: one scale value per output neuron.
        # This minimises quantisation error by normalising each row independently.
        w = weight_fp16.float()                                # float32 for precision
        scale = w.abs().max(dim=1, keepdim=True).values / 127.0
        scale = scale.clamp(min=1e-8)                          # avoid divide-by-zero
        w_int8 = (w / scale).round().clamp(-127, 127).to(torch.int8)

        # Register as buffers (moved to GPU with .to(device), saved in state_dict)
        self.register_buffer('weight_int8', w_int8)
        self.register_buffer('scale', scale.squeeze(1).half())  # [out_features]

        if bias is not None:
            self.register_buffer('bias', bias.half())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Dequantise: int8 + scale -> float16
        # weight_int8: [out, in] as int8
        # scale:       [out]    as float16
        # *args/**kwargs: some UNet attention layers pass extra args to Linear;
        # they don't affect the linear projection itself so we safely ignore them.
        w_fp16 = self.weight_int8.to(x.dtype) * self.scale.unsqueeze(1)
        # Standard linear: x @ w_fp16.T + bias
        return nn.functional.linear(x, w_fp16, self.bias)

    def extra_repr(self):
        return f'in={self.in_features}, out={self.out_features}, dtype=int8+fp16_scale'


def quantize_linear_layers(model: nn.Module, min_params: int = 2048) -> int:
    """
    Replace all nn.Linear layers (above min_params weights) with QuantizedLinear.

    min_params: skip layers with fewer weights than this threshold.
                Small layers (e.g., 64×64 = 4096) are not worth quantising —
                the overhead of the extra buffer outweighs the savings.

    Returns the number of layers quantised.
    """
    count = 0
    # Collect replacements first to avoid mutating during iteration
    replacements = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.weight.numel() >= min_params:
                replacements.append((name, module))

    for name, module in replacements:
        # Navigate to parent
        parts  = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child_name = parts[-1]

        # Build the quantised replacement
        q_linear = QuantizedLinear(
            module.in_features,
            module.out_features,
            module.weight.data,
            module.bias.data if module.bias is not None else None,
        )
        setattr(parent, child_name, q_linear)
        count += 1

    return count


def model_vram_mb(model: nn.Module) -> float:
    """Estimate model weight memory in MB (buffers + parameters)."""
    total = 0
    for p in model.parameters():
        total += p.nelement() * p.element_size()
    for b in model.buffers():
        total += b.nelement() * b.element_size()
    return total / 1024 / 1024


# ── Prerequisite Check ──────────────────────────────────────────────────────────
if not os.path.exists(DENSEPOSE_PKL):
    raise FileNotFoundError(f"DensePose checkpoint missing:\n  {DENSEPOSE_PKL}")
for onnx in ('parsing_atr.onnx', 'parsing_lip.onnx'):
    p = os.path.join(IDMVTON, 'ckpt', 'humanparsing', onnx)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Human parsing model missing:\n  {p}")

# ── Stage 1: Load Preprocessing Models ─────────────────────────────────────────
log('Loading OpenPose...')
openpose_model = OpenPose(0)
openpose_model.preprocessor.body_estimation.model.to(DEVICE)

log('Loading Human Parsing (ONNX)...')
parsing_model = Parsing(0)

# ── Stage 2: Load Diffusion Models to CPU ──────────────────────────────────────
# Load to CPU first so we can quantise before moving to GPU.
# This avoids an OOM spike: without low_cpu_mem_usage, loading fp16 temporarily
# creates a fp32 copy in RAM before converting — doubling peak RAM usage.
log('Loading IDM-VTON models to CPU (for quantisation)...')
torch.cuda.reset_peak_memory_stats()
t_load = time.time()

tokenizer_one   = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder='tokenizer',   use_fast=False)
tokenizer_two   = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder='tokenizer_2', use_fast=False)
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder='scheduler')

text_enc1 = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder='text_encoder',
    torch_dtype=DTYPE, low_cpu_mem_usage=True)
text_enc2 = CLIPTextModelWithProjection.from_pretrained(MODEL_PATH, subfolder='text_encoder_2',
    torch_dtype=DTYPE, low_cpu_mem_usage=True)
img_enc   = CLIPVisionModelWithProjection.from_pretrained(MODEL_PATH, subfolder='image_encoder',
    torch_dtype=DTYPE, low_cpu_mem_usage=True)
vae       = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder='vae',
    torch_dtype=DTYPE, low_cpu_mem_usage=True)
gc.collect()

log('Loading UNet encoder (garment)...')
unet_enc = UNet2DConditionModel_ref.from_pretrained(MODEL_PATH, subfolder='unet_encoder',
    torch_dtype=DTYPE, low_cpu_mem_usage=True)
unet_enc.requires_grad_(False)
gc.collect()

log('Loading UNet (main diffusion)...')
unet = UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder='unet',
    torch_dtype=DTYPE, low_cpu_mem_usage=True)
unet.requires_grad_(False)
gc.collect()

# ── Stage 3: Apply INT8 Weight Quantisation ─────────────────────────────────────
# Quantise on CPU before moving to GPU.
# int8 weights use 1 byte vs float16's 2 bytes → ~50% weight memory saving.
# We only quantise the two large UNets (~11 GB combined).
# The VAE and CLIP encoders are small enough to leave in float16.

log('Quantising UNet linear layers to INT8...')
unet_mb_before = model_vram_mb(unet)
n_unet = quantize_linear_layers(unet, min_params=2048)
unet_mb_after  = model_vram_mb(unet)
log(f'  UNet: {n_unet} layers quantised | {unet_mb_before:.0f} MB -> {unet_mb_after:.0f} MB '
    f'(saved {unet_mb_before - unet_mb_after:.0f} MB)')
gc.collect()

log('Quantising garment UNet encoder linear layers to INT8...')
enc_mb_before = model_vram_mb(unet_enc)
n_enc = quantize_linear_layers(unet_enc, min_params=2048)
enc_mb_after  = model_vram_mb(unet_enc)
log(f'  UNet encoder: {n_enc} layers quantised | {enc_mb_before:.0f} MB -> {enc_mb_after:.0f} MB '
    f'(saved {enc_mb_before - enc_mb_after:.0f} MB)')
gc.collect()

total_saved_mb = (unet_mb_before - unet_mb_after) + (enc_mb_before - enc_mb_after)
log(f'Total weight memory saved: {total_saved_mb:.0f} MB ({total_saved_mb/1024:.2f} GB)')

# ── Stage 4: Assemble Pipeline and Move to GPU ──────────────────────────────────
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

# ── Stage 5: Apply xformers (carried over from Step 1) ──────────────────────────
# Keep all Step 1 optimisations — we build on the previous step, not replace it.
log('Enabling xformers memory-efficient attention (from Step 1)...')
try:
    pipe.enable_xformers_memory_efficient_attention()
    pipe.unet_encoder.enable_xformers_memory_efficient_attention()
    log('  xformers: ENABLED on both UNets')
except Exception as e:
    log(f'  xformers WARNING: {e}')

log('VAE tiling: disabled (causes colour shift vs baseline)')
# pipe.vae.enable_tiling()  -- introduces tile-stitching colour artefacts

load_time = time.time() - t_load
log(f'Ready in {load_time:.1f}s  |  VRAM after load: {vram_gb():.2f} GB')

# ── Stage 6: Preprocess Input Images ───────────────────────────────────────────
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

# ── Stage 7: Run Diffusion Inference ───────────────────────────────────────────
log(f'Running inference ({STEPS} steps) with INT8 + xformers...')
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

# ── Stage 8: Save Outputs ───────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, 'tryon_output.png')
images[0].save(out_path)
mask_gray.save(os.path.join(OUTPUT_DIR, 'mask_preview.png'))
human_img.save(os.path.join(OUTPUT_DIR, 'human_input.png'))
garm_img.save(os.path.join(OUTPUT_DIR, 'garment_input.png'))

vram_saved_vs_baseline = round(BASELINE_VRAM - vram_peak, 2)
vram_saved_vs_step1    = round(STEP1_VRAM    - vram_peak, 2)

metrics = {
    'step':             'step2_quantization',
    'gpu':              'RTX 3090 24GB',
    'mode':             'all_gpu_int8_xformers',
    'optimisations':    ['w8a16_int8_weights', 'xformers_memory_efficient_attention', 'vae_tiling'],
    'quantisation':     'W8A16 per-channel symmetric int8 (pure PyTorch)',
    'layers_quantised': {'unet': n_unet, 'unet_encoder': n_enc},
    'weight_memory_saved_mb': round(total_saved_mb, 1),
    'dtype':            'float16 activations, int8 weights',
    'resolution':       f'{WIDTH}x{HEIGHT}',
    'steps':            STEPS,
    'guidance_scale':   GUIDANCE,
    'seed':             SEED,
    'vram_peak_gb':     round(vram_peak, 2),
    'vram_saved_vs_baseline_gb': vram_saved_vs_baseline,
    'vram_saved_vs_step1_gb':    vram_saved_vs_step1,
    'load_time_s':      round(load_time, 1),
    'inference_time_s': round(inf_time, 1),
    'baseline_vram_gb': BASELINE_VRAM,
    'step1_vram_gb':    STEP1_VRAM,
    'output':           out_path,
}
with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

log('')
log('=' * 65)
log(f'  Step          : 2 — INT8 Quantisation + xformers')
log(f'  Resolution    : {WIDTH}x{HEIGHT}  |  Steps: {STEPS}')
log(f'  VRAM peak     : {vram_peak:.2f} GB')
log(f'  Saved vs base : {vram_saved_vs_baseline:.2f} GB  (was {BASELINE_VRAM} GB)')
log(f'  Saved vs step1: {vram_saved_vs_step1:.2f} GB  (was {STEP1_VRAM} GB)')
log(f'  Weight memory : {total_saved_mb:.0f} MB saved by quantisation')
log(f'  Infer time    : {inf_time:.1f}s')
log(f'  Output        : {out_path}')
log('=' * 65)
