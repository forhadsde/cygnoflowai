"""
IDM-VTON Baseline Inference - Manual CPU offloading (simulates 8 GB VRAM)
Each large model swaps CPU<->GPU per denoising step. Same images as tryon_3090.py.
Expected: ~160s inference vs ~23s for all-GPU.

Run from repo root:
    conda run -n idm python tryon_baseline_windows.py
"""

import sys, os, time, json, gc
from pathlib import Path

ROOT     = str(Path(__file__).parent.resolve())
IDMVTON  = os.path.join(ROOT, 'IDM-VTON')
DEMO_DIR = os.path.join(IDMVTON, 'gradio_demo')

sys.path.insert(0, IDMVTON)
sys.path.insert(0, DEMO_DIR)

import torch
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

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = 'yisol/IDM-VTON'
GARMENT_IMG  = os.path.join(DEMO_DIR, 'example', 'cloth', '04469_00.jpg')
HUMAN_IMG    = os.path.join(DEMO_DIR, 'example', 'human', '00034_00.jpg')
OUTPUT_DIR   = os.path.join(ROOT, 'results', 'baseline')
GARMENT_DESC = 'shirt'
WIDTH, HEIGHT = 768, 1024
STEPS        = 30
GUIDANCE     = 2.0
SEED         = 42
DTYPE        = torch.float16
DEVICE       = 'cuda:0'

DENSEPOSE_CFG = os.path.join(IDMVTON, 'configs', 'densepose_rcnn_R_50_FPN_s1x.yaml')
DENSEPOSE_PKL = os.path.join(IDMVTON, 'ckpt', 'densepose', 'model_final_162be9.pkl')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg): print(f'[BASELINE] {msg}', flush=True)

def vram_gb():
    return torch.cuda.max_memory_allocated() / 1024**3

def gpu(model):
    model.to(DEVICE)
    torch.cuda.empty_cache()
    return model

def cpu(model):
    model.to('cpu')
    torch.cuda.empty_cache()
    gc.collect()
    return model

tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# ── Verify prerequisites ────────────────────────────────────────────────────────
if not os.path.exists(DENSEPOSE_PKL):
    raise FileNotFoundError(f"DensePose checkpoint missing:\n  {DENSEPOSE_PKL}")
for onnx in ('parsing_atr.onnx', 'parsing_lip.onnx'):
    p = os.path.join(IDMVTON, 'ckpt', 'humanparsing', onnx)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Human parsing model missing:\n  {p}")

# ── Load preprocessing ──────────────────────────────────────────────────────────
log('Loading OpenPose...')
openpose_model = OpenPose(0)
openpose_model.preprocessor.body_estimation.model.to(DEVICE)

log('Loading human parsing (ONNX)...')
parsing_model = Parsing(0)

# ── Load all models to CPU (low_cpu_mem_usage avoids fp32 copy RAM spike) ──────
log('Loading IDM-VTON models to CPU...')
torch.cuda.reset_peak_memory_stats()
t_load = time.time()

tokenizer_one   = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder='tokenizer',   use_fast=False)
tokenizer_two   = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder='tokenizer_2', use_fast=False)
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder='scheduler')

text_enc1 = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder='text_encoder',   torch_dtype=DTYPE, low_cpu_mem_usage=True)
text_enc2 = CLIPTextModelWithProjection.from_pretrained(MODEL_PATH, subfolder='text_encoder_2', torch_dtype=DTYPE, low_cpu_mem_usage=True)
img_enc   = CLIPVisionModelWithProjection.from_pretrained(MODEL_PATH, subfolder='image_encoder', torch_dtype=DTYPE, low_cpu_mem_usage=True)
vae       = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder='vae', torch_dtype=DTYPE, low_cpu_mem_usage=True)
gc.collect()

log('Loading UNet encoder (garment)...')
unet_enc = UNet2DConditionModel_ref.from_pretrained(MODEL_PATH, subfolder='unet_encoder', torch_dtype=DTYPE, low_cpu_mem_usage=True)
unet_enc.requires_grad_(False)
gc.collect()

log('Loading UNet (main diffusion)...')
unet = UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder='unet', torch_dtype=DTYPE, low_cpu_mem_usage=True)
unet.requires_grad_(False)
gc.collect()

load_time = time.time() - t_load
log(f'All models loaded in {load_time:.1f}s  |  VRAM: {vram_gb():.2f} GB')

# ── Prepare input images ────────────────────────────────────────────────────────
log('Preprocessing inputs...')
garm_img  = Image.open(GARMENT_IMG).convert('RGB').resize((WIDTH, HEIGHT))
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

# ── Step 1: Text encoding ───────────────────────────────────────────────────────
log('Step 1/5: Text encoding...')
gpu(text_enc1); gpu(text_enc2)

prompt     = f'model is wearing {GARMENT_DESC}'
neg_prompt = 'monochrome, lowres, bad anatomy, worst quality, low quality'
prompt_c   = f'a photo of {GARMENT_DESC}'

def encode_text(prompt_list, tokenizer, encoder):
    tokens = tokenizer(
        prompt_list, padding='max_length', max_length=tokenizer.model_max_length,
        truncation=True, return_tensors='pt',
    )
    with torch.no_grad():
        return encoder(tokens.input_ids.to(DEVICE), output_hidden_states=True)

out1_pos  = encode_text([prompt],     tokenizer_one, text_enc1)
out1_neg  = encode_text([neg_prompt], tokenizer_one, text_enc1)
out2_pos  = encode_text([prompt],     tokenizer_two, text_enc2)
out2_neg  = encode_text([neg_prompt], tokenizer_two, text_enc2)
out1_cpos = encode_text([prompt_c],   tokenizer_one, text_enc1)
out2_cpos = encode_text([prompt_c],   tokenizer_two, text_enc2)

prompt_embeds     = out1_pos.hidden_states[-2].to(DTYPE)
neg_embeds        = out1_neg.hidden_states[-2].to(DTYPE)
pooled_pos        = out2_pos[0].to(DTYPE)
pooled_neg        = out2_neg[0].to(DTYPE)
text_embeds_cloth = torch.cat([
    out1_cpos.hidden_states[-2].to(DTYPE),
    out2_cpos.hidden_states[-2].to(DTYPE),
], dim=-1)

prompt_embeds2 = out2_pos.hidden_states[-2].to(DTYPE)
neg_embeds2    = out2_neg.hidden_states[-2].to(DTYPE)
prompt_embeds  = torch.cat([prompt_embeds, prompt_embeds2], dim=-1)
neg_embeds     = torch.cat([neg_embeds, neg_embeds2], dim=-1)

cpu(text_enc1); cpu(text_enc2)
del text_enc1, text_enc2
gc.collect()

# ── Step 2: Image (ip-adapter) encoding ────────────────────────────────────────
log('Step 2/5: Image encoding...')
gpu(img_enc)
clip_proc = CLIPImageProcessor()
garm_pix  = clip_proc(images=garm_img, return_tensors='pt').pixel_values.to(DEVICE, DTYPE)
with torch.no_grad():
    img_emb_out = img_enc(garm_pix, output_hidden_states=True)
    img_embeds  = img_emb_out.hidden_states[-2]
cpu(img_enc)
del img_enc
gc.collect()

# encoder_hid_proj is one linear layer — run on GPU (CPU fp16 matmul unsupported on Windows)
with torch.no_grad():
    unet.encoder_hid_proj.to(DEVICE)
    ip_image_embeds = unet.encoder_hid_proj(img_embeds.to(DTYPE)).cpu()
    unet.encoder_hid_proj.to('cpu')
ip_image_embeds = ip_image_embeds.to(DTYPE)
ip_image_embeds = torch.cat([ip_image_embeds] * 2)

# ── Step 3: VAE encoding ────────────────────────────────────────────────────────
log('Step 3/5: VAE encoding...')
gpu(vae)
vae_scale = vae.config.scaling_factor

def vae_encode(pil_img):
    t = tensor_transform(pil_img).unsqueeze(0).to(DEVICE, DTYPE)
    with torch.no_grad():
        return vae.encode(t).latent_dist.sample() * vae_scale

cloth_lat  = vae_encode(garm_img)
human_lat  = vae_encode(human_img)
pose_lat   = vae_encode(pose_img)

mask_t       = transforms.ToTensor()(mask).unsqueeze(0).to(DEVICE, DTYPE)
mask_lat     = torch.nn.functional.interpolate(mask_t, size=(HEIGHT//8, WIDTH//8))
masked_human = human_lat * (1 - mask_lat)
cpu(vae)
del vae
gc.collect()

# ── Step 4: Denoising loop (CPU<->GPU swap each step) ───────────────────────────
log('Step 4/5: Denoising (30 steps, CPU offloading per step)...')
torch.cuda.reset_peak_memory_stats()
t_inf = time.time()

noise_scheduler.set_timesteps(STEPS, device='cpu')
timesteps = noise_scheduler.timesteps

generator = torch.Generator('cpu').manual_seed(SEED)
latents   = torch.randn((1, 4, HEIGHT // 8, WIDTH // 8), generator=generator, dtype=DTYPE)
latents   = latents * noise_scheduler.init_noise_sigma

add_time_ids = torch.tensor([[HEIGHT, WIDTH, 0, 0, HEIGHT, WIDTH]], dtype=DTYPE)

for i, t in enumerate(timesteps):
    log(f'  Step {i+1}/{STEPS}...')

    # garment encoder: CPU -> GPU -> CPU
    gpu(unet_enc)
    with torch.no_grad():
        _, ref_feats = unet_enc(
            cloth_lat.to(DEVICE), t.to(DEVICE),
            text_embeds_cloth.to(DEVICE), return_dict=False,
        )
    ref_feats = [torch.cat([torch.zeros_like(f.cpu()), f.cpu()]) for f in ref_feats]
    cpu(unet_enc)

    # main UNet: CPU -> GPU -> CPU
    gpu(unet)
    lat_in    = noise_scheduler.scale_model_input(torch.cat([latents] * 2).to(DEVICE, DTYPE), t)
    lat_in    = torch.cat([
        lat_in,
        torch.cat([mask_lat] * 2).to(DEVICE, DTYPE),
        torch.cat([masked_human] * 2).to(DEVICE, DTYPE),
        torch.cat([pose_lat] * 2).to(DEVICE, DTYPE),
    ], dim=1)

    with torch.no_grad():
        noise_pred = unet(
            lat_in, t.to(DEVICE),
            encoder_hidden_states=torch.cat([neg_embeds, prompt_embeds]).to(DEVICE, DTYPE),
            added_cond_kwargs={
                'text_embeds':  torch.cat([pooled_neg, pooled_pos]).to(DEVICE, DTYPE),
                'time_ids':     torch.cat([add_time_ids] * 2).to(DEVICE, DTYPE),
                'image_embeds': ip_image_embeds.to(DEVICE, DTYPE),
            },
            garment_features=[f.to(DEVICE, DTYPE) for f in ref_feats],
            return_dict=False,
        )[0]
    cpu(unet)

    noise_uncond, noise_text = noise_pred.cpu().chunk(2)
    latents = noise_scheduler.step(
        noise_uncond + GUIDANCE * (noise_text - noise_uncond), t, latents
    ).prev_sample

inf_time  = time.time() - t_inf
vram_peak = vram_gb()
log(f'Denoising done: {inf_time:.1f}s  |  VRAM peak: {vram_peak:.2f} GB')

# ── Step 5: VAE decode ──────────────────────────────────────────────────────────
log('Step 5/5: Decoding image...')
del unet, unet_enc
gc.collect()
vae = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder='vae', torch_dtype=DTYPE, low_cpu_mem_usage=True)
vae_scale = vae.config.scaling_factor
gpu(vae)
with torch.no_grad():
    decoded = vae.decode(latents.to(DEVICE, DTYPE) / vae_scale).sample
cpu(vae)
decoded = (decoded / 2 + 0.5).clamp(0, 1)
out_img = transforms.ToPILImage()(decoded.squeeze(0).float().cpu())

# ── Save outputs ────────────────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, 'tryon_output.png')
out_img.save(out_path)
mask_gray.save(os.path.join(OUTPUT_DIR, 'mask_preview.png'))
human_img.save(os.path.join(OUTPUT_DIR, 'human_input.png'))
garm_img.save(os.path.join(OUTPUT_DIR, 'garment_input.png'))

metrics = {
    'gpu':              'RTX 3090 24GB (CPU offload, simulates 8 GB)',
    'mode':             'cpu_offload_per_step',
    'dtype':            'float16',
    'resolution':       f'{WIDTH}x{HEIGHT}',
    'steps':            STEPS,
    'guidance_scale':   GUIDANCE,
    'cpu_offload':      'manual_alternating_unet_enc',
    'vram_peak_gb':     round(vram_peak, 2),
    'load_time_s':      round(load_time, 1),
    'inference_time_s': round(inf_time, 1),
    'output':           out_path,
}
with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

log('')
log('=' * 55)
log(f'  Output     : {out_path}')
log(f'  Resolution : {WIDTH}x{HEIGHT}  |  Steps: {STEPS}')
log(f'  VRAM peak  : {vram_peak:.2f} GB')
log(f'  Load time  : {load_time:.1f}s')
log(f'  Infer time : {inf_time:.1f}s')
log('=' * 55)
