"""
IDM-VTON Baseline Inference  —  Manual device management for 8 GB VRAM
-----------------------------------------------------------------------
FP16, 768x1024, 30 steps.
Each large model (unet ~6 GB, unet_encoder ~5 GB) loads directly to GPU
when needed and immediately offloads to CPU afterwards — no double-copy OOM.

Run from: /home/rey/cygnoflow/
"""

import sys, os, time, json, gc
ROOT = '/home/rey/cygnoflow'
sys.path.insert(0, f'{ROOT}/IDM-VTON')
sys.path.insert(0, f'{ROOT}/IDM-VTON/gradio_demo')

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

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = f'{ROOT}/IDM-VTON/ckpt/IDM-VTON'
GARMENT_IMG  = f'{ROOT}/jacketfrontside.png'
HUMAN_IMG    = f'{ROOT}/model1.jpg'
OUTPUT_DIR   = f'{ROOT}/results/baseline'
GARMENT_DESC = 'jacket'
WIDTH, HEIGHT = 768, 1024
STEPS        = 30
GUIDANCE     = 2.0
SEED         = 42
DTYPE        = torch.float16
DEVICE       = 'cuda:0'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(f'{ROOT}/IDM-VTON/gradio_demo')

def log(msg): print(f'[TRYON] {msg}', flush=True)

def vram_gb():
    return torch.cuda.max_memory_allocated() / 1024**3

def gpu(model):
    """Move model to GPU with empty_cache after."""
    model.to(DEVICE)
    torch.cuda.empty_cache()
    return model

def cpu(model):
    """Move model to CPU and free VRAM."""
    model.to('cpu')
    torch.cuda.empty_cache()
    gc.collect()
    return model

tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# ── Load preprocessing ────────────────────────────────────────────────────────
log('Loading OpenPose...')
openpose_model = OpenPose(0)
openpose_model.preprocessor.body_estimation.model.to(DEVICE)

log('Loading human parsing...')
parsing_model = Parsing(0)

# ── Load diffusion models (all on CPU initially) ───────────────────────────────
# low_cpu_mem_usage=True avoids the temporary fp32 copy when loading fp16 weights,
# preventing the ~2x RAM spike that OOM-kills on 15 GB RAM systems.
log('Loading IDM-VTON models to CPU...')
torch.cuda.reset_peak_memory_stats()
t_load = time.time()

tokenizer_one  = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder='tokenizer',   use_fast=False)
tokenizer_two  = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder='tokenizer_2', use_fast=False)
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

# ── Prepare input images ──────────────────────────────────────────────────────
log('Preprocessing inputs...')
garm_img  = Image.open(GARMENT_IMG).convert('RGB').resize((WIDTH, HEIGHT))
human_img = Image.open(HUMAN_IMG).convert('RGB').resize((WIDTH, HEIGHT))

keypoints   = openpose_model(human_img.resize((384, 512)))
model_parse, _ = parsing_model(human_img.resize((384, 512)))
mask, mask_gray = get_mask_location('hd', 'upper_body', model_parse, keypoints)
mask = mask.resize((WIDTH, HEIGHT))
mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

human_arg = _apply_exif_orientation(human_img.resize((384, 512)))
human_arg = convert_PIL_to_numpy(human_arg, format='BGR')
dp_args = apply_net.create_argument_parser().parse_args((
    'show',
    f'{ROOT}/IDM-VTON/configs/densepose_rcnn_R_50_FPN_s1x.yaml',
    f'{ROOT}/IDM-VTON/ckpt/densepose/model_final_162be9.pkl',
    'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda',
))
pose_img = dp_args.func(dp_args, human_arg)[:, :, ::-1]
pose_img = Image.fromarray(pose_img).resize((WIDTH, HEIGHT))

# ── Step 1: Text encoding (small models on GPU, back to CPU after) ─────────────
log('Step 1/5: Text encoding...')
gpu(text_enc1); gpu(text_enc2)

prompt      = f'model is wearing {GARMENT_DESC}'
neg_prompt  = 'monochrome, lowres, bad anatomy, worst quality, low quality'
prompt_c    = f'a photo of {GARMENT_DESC}'

def encode_text(prompt_list, tokenizer, encoder, add_special=True):
    tokens = tokenizer(
        prompt_list, padding='max_length', max_length=tokenizer.model_max_length,
        truncation=True, return_tensors='pt',
    )
    with torch.no_grad():
        out = encoder(tokens.input_ids.to(DEVICE), output_hidden_states=True)
    return out

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
# Garment encoder cross-attn expects 2048-dim = 768 (enc1) + 1280 (enc2) concatenated
text_embeds_cloth = torch.cat([
    out1_cpos.hidden_states[-2].to(DTYPE),
    out2_cpos.hidden_states[-2].to(DTYPE),
], dim=-1)

# Concat CLIP outputs for SDXL
prompt_embeds2 = out2_pos.hidden_states[-2].to(DTYPE)
neg_embeds2    = out2_neg.hidden_states[-2].to(DTYPE)
prompt_embeds  = torch.cat([prompt_embeds, prompt_embeds2], dim=-1)
neg_embeds     = torch.cat([neg_embeds, neg_embeds2], dim=-1)

cpu(text_enc1); cpu(text_enc2)
# Free text encoders from RAM — no longer needed after embeddings are extracted
del text_enc1, text_enc2
gc.collect()

# ── Step 2: Image (ip-adapter) encoding ───────────────────────────────────────
log('Step 2/5: Image encoding...')
gpu(img_enc)
clip_proc = CLIPImageProcessor()
garm_pix  = clip_proc(images=garm_img, return_tensors='pt').pixel_values.to(DEVICE, DTYPE)
with torch.no_grad():
    img_emb_out = img_enc(garm_pix, output_hidden_states=True)
    img_embeds = img_emb_out.hidden_states[-2]
cpu(img_enc)
# Free image encoder from RAM — no longer needed
del img_enc
gc.collect()

# Project via unet's encoder_hid_proj (small, leave on CPU and project there)
with torch.no_grad():
    ip_image_embeds = unet.encoder_hid_proj(img_embeds.cpu().to(dtype=torch.float16))
ip_image_embeds = ip_image_embeds.to(DTYPE)
# For CFG: double the embeddings
ip_image_embeds = torch.cat([ip_image_embeds] * 2)   # [neg, pos] for CFG

# ── Step 3: VAE encoding ──────────────────────────────────────────────────────
log('Step 3/5: VAE encoding...')
gpu(vae)
vae_scale = vae.config.scaling_factor

def vae_encode(pil_img):
    t = tensor_transform(pil_img).unsqueeze(0).to(DEVICE, DTYPE)
    with torch.no_grad():
        lat = vae.encode(t).latent_dist.sample() * vae_scale
    return lat

cloth_lat  = vae_encode(garm_img)
human_lat  = vae_encode(human_img)
pose_lat   = vae_encode(pose_img)   # 4-ch VAE latent, H/8 × W/8

# Mask + masked image latents
mask_t = transforms.ToTensor()(mask).unsqueeze(0).to(DEVICE, DTYPE)
mask_lat = torch.nn.functional.interpolate(mask_t, size=(HEIGHT//8, WIDTH//8))
masked_human = human_lat * (1 - mask_lat)
cpu(vae)
# Free VAE from RAM during the denoising loop; will reload for decode
del vae
gc.collect()

# ── Step 4: Denoising loop ────────────────────────────────────────────────────
log('Step 4/5: Denoising (30 steps)...')
torch.cuda.reset_peak_memory_stats()
t_inf = time.time()

noise_scheduler.set_timesteps(STEPS, device='cpu')
timesteps = noise_scheduler.timesteps

generator = torch.Generator('cpu').manual_seed(SEED)
latents   = torch.randn(
    (1, 4, HEIGHT // 8, WIDTH // 8),
    generator=generator, dtype=DTYPE,
)
latents   = latents * noise_scheduler.init_noise_sigma

# Prepare time ids (SDXL)
add_time_ids = torch.tensor([[HEIGHT, WIDTH, 0, 0, HEIGHT, WIDTH]], dtype=DTYPE)

for i, t in enumerate(timesteps):
    log(f'  Step {i+1}/{STEPS}...')

    # ── 4a: garment encoder (CPU → GPU → CPU) ─────────────────────────────────
    gpu(unet_enc)
    cloth_gpu = cloth_lat.to(DEVICE)
    t_gpu     = t.to(DEVICE)
    tc_gpu    = text_embeds_cloth.to(DEVICE)
    with torch.no_grad():
        _, ref_feats = unet_enc(cloth_gpu, t_gpu, tc_gpu, return_dict=False)
    ref_feats = list(ref_feats)
    # CFG doubling on CPU
    ref_feats = [torch.cat([torch.zeros_like(f.cpu()), f.cpu()]) for f in ref_feats]
    cpu(unet_enc)

    # ── 4b: main UNet denoising step (CPU → GPU → CPU) ────────────────────────
    gpu(unet)
    lat_in   = torch.cat([latents] * 2).to(DEVICE, DTYPE)
    lat_in   = noise_scheduler.scale_model_input(lat_in, t)
    # 13-channel input: noisy latent + mask + masked_image + pose
    mask_in   = torch.cat([mask_lat] * 2).to(DEVICE, DTYPE)
    masked_in = torch.cat([masked_human] * 2).to(DEVICE, DTYPE)
    pose_in   = torch.cat([pose_lat] * 2).to(DEVICE, DTYPE)
    lat_in    = torch.cat([lat_in, mask_in, masked_in, pose_in], dim=1)

    add_cond = {
        'text_embeds': torch.cat([pooled_neg, pooled_pos]).to(DEVICE, DTYPE),
        'time_ids':    torch.cat([add_time_ids] * 2).to(DEVICE, DTYPE),
        'image_embeds': ip_image_embeds.to(DEVICE, DTYPE),
    }
    ref_feats_gpu = [f.to(DEVICE, DTYPE) for f in ref_feats]
    enc_hidden = torch.cat([neg_embeds, prompt_embeds]).to(DEVICE, DTYPE)

    with torch.no_grad():
        noise_pred = unet(
            lat_in, t.to(DEVICE),
            encoder_hidden_states=enc_hidden,
            added_cond_kwargs=add_cond,
            garment_features=ref_feats_gpu,
            return_dict=False,
        )[0]
    cpu(unet)

    # Classifier-free guidance
    noise_uncond, noise_text = noise_pred.chunk(2)
    noise_pred = noise_uncond + GUIDANCE * (noise_text - noise_uncond)

    latents = noise_scheduler.step(noise_pred.cpu(), t, latents).prev_sample

inf_time  = time.time() - t_inf
vram_peak = vram_gb()
log(f'  Denoising done: {inf_time:.1f}s  |  VRAM peak: {vram_peak:.2f} GB')

# ── Step 5: VAE decode ────────────────────────────────────────────────────────
log('Step 5/5: Decoding image...')
# Reload VAE (was freed before the denoising loop to save RAM)
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

# ── Save outputs ──────────────────────────────────────────────────────────────
out_path = f'{OUTPUT_DIR}/tryon_output.png'
out_img.save(out_path)
mask_gray.save(f'{OUTPUT_DIR}/mask_preview.png')
human_img.save(f'{OUTPUT_DIR}/human_input.png')
garm_img.save(f'{OUTPUT_DIR}/garment_input.png')

metrics = {
    'mode':              'baseline',
    'dtype':             'float16',
    'resolution':        f'{WIDTH}x{HEIGHT}',
    'steps':             STEPS,
    'guidance_scale':    GUIDANCE,
    'attention_slicing': False,
    'vae_slicing':       False,
    'cpu_offload':       'manual_alternating',
    'vram_peak_gb':      round(vram_peak, 2),
    'load_time_s':       round(load_time, 1),
    'inference_time_s':  round(inf_time, 1),
    'output':            out_path,
}
with open(f'{OUTPUT_DIR}/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

log('')
log('═' * 55)
log(f'  Output     : {out_path}')
log(f'  Resolution : {WIDTH}x{HEIGHT}  |  Steps: {STEPS}')
log(f'  Precision  : float16')
log(f'  VRAM peak  : {vram_peak:.2f} GB')
log(f'  Load time  : {load_time:.1f}s')
log(f'  Infer time : {inf_time:.1f}s')
log('═' * 55)
