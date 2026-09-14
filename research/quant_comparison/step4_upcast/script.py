"""
QUANT COMPARISON -- Step 4: INT8 + xformers + upcast_attention
===============================================================
Tests whether upcast_attention=True fixes the colour shift introduced by xformers.

Root cause of xformers colour shift:
  FlashAttention reorders float16 additions for memory efficiency.
  Float16 is not associative: (a+b)+c != a+(b+c).
  Tiny per-step errors compound over 30 denoising steps -> visible colour drift.

Hypothesis:
  upcast_attention=True forces the attention score matrix to be computed in
  float32, then cast back to float16 before softmax. The accumulation happens
  in float32 where rounding errors are negligible, so xformers can reorder
  float16 I/O without introducing compounding errors.

If this works: VRAM = ~10 GB (xformers) with colour accuracy preserved.
If it doesn't:  colour drift persists -> xformers is fundamentally incompatible.

Config: INT8 weights + xformers + upcast_attention=True
"""

import sys, os, time, json
from pathlib import Path

ROOT     = str(Path(__file__).parent.parent.parent.parent.resolve())
IDMVTON  = os.path.join(ROOT, 'IDM-VTON')
DEMO_DIR = os.path.join(IDMVTON, 'gradio_demo')
sys.path.insert(0, IDMVTON)
sys.path.insert(0, DEMO_DIR)

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import DDPMScheduler, AutoencoderKL
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from utils_mask import get_mask_location
import apply_net
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

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

os.makedirs(OUTPUT_DIR, exist_ok=True)
def log(msg): print(f'[UPCAST] {msg}', flush=True)
def vram_gb(): return torch.cuda.max_memory_allocated() / 1024**3

tensor_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

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

# ── INT8 Implementation ─────────────────────────────────────────────────────────
class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, weight_fp16, bias=None):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        w = weight_fp16.float()
        scale = w.abs().max(dim=1, keepdim=True).values / 127.0
        scale = scale.clamp(min=1e-8)
        w_int8 = (w / scale).round().clamp(-127, 127).to(torch.int8)
        self.register_buffer('weight_int8', w_int8)
        self.register_buffer('scale', scale.squeeze(1).half())
        if bias is not None:
            self.register_buffer('bias', bias.half())
        else:
            self.bias = None

    def forward(self, x, *args, **kwargs):
        w_fp16 = self.weight_int8.to(x.dtype) * self.scale.unsqueeze(1)
        return nn.functional.linear(x, w_fp16, self.bias)

def quantize_linear_layers(model, min_params=2048):
    replaced, before_mb = 0, sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or module.weight.numel() < min_params:
            continue
        parts = name.rsplit('.', 1)
        parent = model
        for seg in (parts[:-1][0].split('.') if len(parts) > 1 else []):
            parent = getattr(parent, seg)
        child_name = parts[-1] if len(parts) > 1 else name
        setattr(parent, child_name, QuantizedLinear(
            module.in_features, module.out_features,
            module.weight.data, module.bias.data if module.bias is not None else None))
        replaced += 1
    after_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    after_mb += sum(b.numel() * b.element_size() for b in model.buffers()) / 1024**2
    return replaced, round(before_mb), round(after_mb)

# ── Load Preprocessing Models ───────────────────────────────────────────────────
log('Loading OpenPose...')
openpose_model = OpenPose(0)
openpose_model.preprocessor.body_estimation.model.to(DEVICE)
log('Loading Human Parsing...')
parsing_model = Parsing(0)

# ── Load UNets with upcast_attention=True ──────────────────────────────────────
log('Loading models to CPU with upcast_attention=True...')
log('  upcast_attention forces attention scores to float32 before softmax')
log('  This prevents float16 non-associativity errors that cause colour drift with xformers')
torch.cuda.reset_peak_memory_stats()
t_load = time.time()

tokenizer_one   = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder='tokenizer',   use_fast=False)
tokenizer_two   = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder='tokenizer_2', use_fast=False)
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder='scheduler')
text_enc1 = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder='text_encoder',   torch_dtype=DTYPE)
text_enc2 = CLIPTextModelWithProjection.from_pretrained(MODEL_PATH, subfolder='text_encoder_2', torch_dtype=DTYPE)
img_enc   = CLIPVisionModelWithProjection.from_pretrained(MODEL_PATH, subfolder='image_encoder', torch_dtype=DTYPE)
vae       = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder='vae', torch_dtype=DTYPE)

unet_enc  = UNet2DConditionModel_ref.from_pretrained(MODEL_PATH, subfolder='unet_encoder',
                                                      torch_dtype=DTYPE, upcast_attention=True)
unet_enc.requires_grad_(False)

unet      = UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder='unet',
                                                  torch_dtype=DTYPE, upcast_attention=True)
unet.requires_grad_(False)

# ── INT8 Quantisation ───────────────────────────────────────────────────────────
log('Quantising both UNets to INT8...')
n,  before,  after  = quantize_linear_layers(unet)
n2, before2, after2 = quantize_linear_layers(unet_enc)
total_saved = (before + before2) - (after + after2)
log(f'  UNet:         {n}  layers | {before} MB -> {after} MB')
log(f'  UNet encoder: {n2} layers | {before2} MB -> {after2} MB')
log(f'  Total weight memory saved: {total_saved} MB ({total_saved/1024:.2f} GB)')

pipe = TryonPipeline.from_pretrained(MODEL_PATH, unet=unet, vae=vae,
    feature_extractor=CLIPImageProcessor(), text_encoder=text_enc1,
    text_encoder_2=text_enc2, tokenizer=tokenizer_one, tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler, image_encoder=img_enc, torch_dtype=DTYPE)
pipe.unet_encoder = unet_enc
pipe.to(DEVICE)
pipe.unet_encoder.to(DEVICE)

# ── Enable xformers ─────────────────────────────────────────────────────────────
log('Enabling xformers memory-efficient attention...')
try:
    pipe.enable_xformers_memory_efficient_attention()
    log('  xformers: ENABLED on main UNet')
except Exception as e:
    log(f'  xformers on pipe: {e}')
try:
    pipe.unet_encoder.enable_xformers_memory_efficient_attention()
    log('  xformers: ENABLED on UNet encoder')
except Exception as e:
    log(f'  xformers on unet_encoder: {e}')

load_time = time.time() - t_load
log(f'Ready in {load_time:.1f}s  |  VRAM after load: {vram_gb():.2f} GB')

# ── Preprocess Inputs ───────────────────────────────────────────────────────────
log('Preprocessing...')
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
dp_args = apply_net.create_argument_parser().parse_args(('show', DENSEPOSE_CFG, DENSEPOSE_PKL, 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
pose_img = Image.fromarray(dp_args.func(dp_args, human_arg)[:, :, ::-1]).resize((WIDTH, HEIGHT))

# ── Inference ───────────────────────────────────────────────────────────────────
log(f'Running inference ({STEPS} steps) with INT8 + xformers + upcast_attention...')
torch.cuda.reset_peak_memory_stats()
t_inf = time.time()

with torch.no_grad():
    with torch.cuda.amp.autocast():
        with torch.inference_mode():
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                f'model is wearing {GARMENT_DESC}', num_images_per_prompt=1,
                do_classifier_free_guidance=True, negative_prompt='monochrome, lowres, bad anatomy, worst quality, low quality')
        with torch.inference_mode():
            prompt_embeds_c, _, _, _ = pipe.encode_prompt(
                f'a photo of {GARMENT_DESC}', num_images_per_prompt=1,
                do_classifier_free_guidance=False, negative_prompt='')

        pose_img_t  = tensor_transform(pose_img).unsqueeze(0).to(DEVICE, DTYPE)
        garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(DEVICE, DTYPE)
        generator   = torch.Generator(DEVICE).manual_seed(SEED)

        images = pipe(
            prompt_embeds=prompt_embeds.to(DEVICE, DTYPE),
            negative_prompt_embeds=negative_prompt_embeds.to(DEVICE, DTYPE),
            pooled_prompt_embeds=pooled_prompt_embeds.to(DEVICE, DTYPE),
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(DEVICE, DTYPE),
            num_inference_steps=STEPS, generator=generator, strength=1.0,
            pose_img=pose_img_t, text_embeds_cloth=prompt_embeds_c.to(DEVICE, DTYPE),
            cloth=garm_tensor, mask_image=mask, image=human_img,
            height=HEIGHT, width=WIDTH,
            ip_adapter_image=garm_img.resize((WIDTH, HEIGHT)), guidance_scale=GUIDANCE,
        )[0]

inf_time  = time.time() - t_inf
vram_peak = vram_gb()

# ── Save Results ────────────────────────────────────────────────────────────────
images[0].save(os.path.join(OUTPUT_DIR, 'tryon_output.png'))
mask_gray.save(os.path.join(OUTPUT_DIR, 'mask_preview.png'))
human_img.save(os.path.join(OUTPUT_DIR, 'human_input.png'))
garm_img.save(os.path.join(OUTPUT_DIR, 'garment_input.png'))

metrics = {
    'step': 'int8_xformers_upcast',
    'vram_peak_gb': round(vram_peak, 2),
    'inference_time_s': round(inf_time, 1),
    'optimisations': ['int8_weights', 'xformers', 'upcast_attention'],
    'weight_precision': 'int8 weights, float16 activations',
    'attention': 'xformers FlashAttention + upcast_attention=True (float32 accumulation)',
    'layers_quantised': n + n2,
    'weight_saved_mb': total_saved,
    'resolution': f'{WIDTH}x{HEIGHT}',
    'hypothesis': 'upcast_attention fixes xformers colour shift by computing attention in float32',
}
with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

log('')
log('=' * 60)
log(f'  Config        : INT8 + xformers + upcast_attention=True')
log(f'  VRAM peak     : {vram_peak:.2f} GB')
log(f'  Infer time    : {inf_time:.1f}s')
log(f'  Weight saved  : {total_saved} MB ({total_saved/1024:.2f} GB)')
log(f'  Output        : {OUTPUT_DIR}/tryon_output.png')
log('=' * 60)
log('')
log('Compare this output colour vs step0_baseline/results/tryon_output.png')
log('If colour matches: upcast_attention fixes xformers drift -> use this config')
log('If colour drifts:  xformers incompatible even with float32 attn -> stay on INT8 no-xformers')
