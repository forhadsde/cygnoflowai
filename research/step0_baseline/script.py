"""
STEP 0 — Baseline Inference (Annotated)
========================================
Research: Memory Optimisation for Generative AI
Hardware: NVIDIA RTX 3090 (24 GB VRAM)
Resolution: 768 x 1024
Strategy: All models loaded directly to GPU. No memory optimisation.

Result:
  - Inference time : 23.2 seconds
  - Peak VRAM      : 16.29 GB
  - Quality        : Full quality (this is the reference output)

Why this is the baseline:
  Loading everything to GPU is the simplest and fastest strategy.
  It requires the most VRAM. All subsequent steps will reduce VRAM
  usage while keeping quality as close as possible to this result.
"""

import sys, os, time, json, gc
from pathlib import Path

# ── Path Setup ──────────────────────────────────────────────────────────────────
# We need to tell Python where to find the IDM-VTON source code.
# ROOT is the cygnoflow repo root (where this research/ folder lives).
ROOT     = str(Path(__file__).parent.parent.parent.resolve())
IDMVTON  = os.path.join(ROOT, 'IDM-VTON')
DEMO_DIR = os.path.join(IDMVTON, 'gradio_demo')

# sys.path tells Python where to look when you write "import something".
# IDM-VTON first: so "from src.tryon_pipeline import ..." resolves here.
# gradio_demo second: so "import apply_net" and detectron2 resolve here.
sys.path.insert(0, IDMVTON)
sys.path.insert(0, DEMO_DIR)

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

# CLIP: Contrastive Language-Image Pre-training — encodes text and images
# into vectors the diffusion model can use as guidance.
from transformers import (
    AutoTokenizer, CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel, CLIPTextModelWithProjection,
)

# diffusers: HuggingFace library for diffusion model pipelines.
# DDPMScheduler: controls the noise schedule (how noise is removed each step).
# AutoencoderKL: the VAE that compresses/decompresses images.
from diffusers import DDPMScheduler, AutoencoderKL

# IDM-VTON's modified pipeline and UNet architectures.
# "hacked" means they've been modified to accept garment conditioning.
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel

# Preprocessing models: body pose detection and body region segmentation.
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

# Utility to determine which body region to replace with the garment.
from utils_mask import get_mask_location

# DensePose: maps each pixel to a 3D body surface coordinate (UV map).
import apply_net
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

# ── Configuration ───────────────────────────────────────────────────────────────
# MODEL_PATH: HuggingFace repo ID. Downloads ~12 GB on first run, then cached.
MODEL_PATH   = 'yisol/IDM-VTON'

# Example images bundled with IDM-VTON for testing.
GARMENT_IMG  = os.path.join(ROOT, 'research', 'denimshirt1.png')
HUMAN_IMG    = os.path.join(DEMO_DIR, 'example', 'human', '00034_00.jpg')

# Results go into this script's own results/ subfolder.
OUTPUT_DIR   = os.path.join(Path(__file__).parent.resolve(), 'results')

GARMENT_DESC = 'blue denim shirt'  # Text description of the garment
WIDTH, HEIGHT = 768, 1024        # Output resolution (SDXL default for portrait)
STEPS        = 30                # Denoising steps (more = slower but better quality)
GUIDANCE     = 2.0               # Classifier-free guidance scale
SEED         = 42                # Fixed seed for reproducibility across experiments
DTYPE        = torch.float16     # float16 uses half the memory of float32
DEVICE       = 'cuda:0'          # First GPU

# DensePose config and checkpoint files (downloaded by setup script).
DENSEPOSE_CFG = os.path.join(IDMVTON, 'configs', 'densepose_rcnn_R_50_FPN_s1x.yaml')
DENSEPOSE_PKL = os.path.join(IDMVTON, 'ckpt', 'densepose', 'model_final_162be9.pkl')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg): print(f'[STEP0] {msg}', flush=True)

def vram_gb():
    """Returns the peak GPU memory allocated so far, in gigabytes."""
    return torch.cuda.max_memory_allocated() / 1024**3

# Standard image normalisation: maps pixel values from [0,255] to [-1,1].
# Diffusion models expect this range.
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

# ── Prerequisite Check ──────────────────────────────────────────────────────────
# Fail early with a clear message if model files are missing.
if not os.path.exists(DENSEPOSE_PKL):
    raise FileNotFoundError(
        f"DensePose checkpoint missing:\n  {DENSEPOSE_PKL}\n"
        "Run setup_idm_windows.ps1 first."
    )
for onnx in ('parsing_atr.onnx', 'parsing_lip.onnx'):
    p = os.path.join(IDMVTON, 'ckpt', 'humanparsing', onnx)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Human parsing model missing:\n  {p}")

# ── Stage 1: Load Preprocessing Models ─────────────────────────────────────────
# These are small models. They run once on the input image before diffusion starts.

log('Loading OpenPose (body pose detector)...')
# OpenPose(0) means "use GPU 0". Detects 18 body keypoints (joints).
openpose_model = OpenPose(0)
openpose_model.preprocessor.body_estimation.model.to(DEVICE)

log('Loading Human Parsing (body region segmenter, ONNX)...')
# Runs via ONNX Runtime GPU. Segments the image into 20 body regions.
parsing_model = Parsing(0)

# ── Stage 2: Load Diffusion Models — All Directly to GPU ───────────────────────
# BASELINE STRATEGY: load everything to GPU immediately.
# Pro: no transfers during inference → fastest possible inference.
# Con: requires ~16 GB VRAM → cannot run on most consumer GPUs.

log('Loading IDM-VTON diffusion models to GPU...')
torch.cuda.reset_peak_memory_stats()   # Reset the VRAM counter to 0
t_load = time.time()

# Tokenisers: convert text strings into token IDs for the CLIP encoders.
# These live in RAM, not VRAM.
tokenizer_one  = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder='tokenizer',   use_fast=False)
tokenizer_two  = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder='tokenizer_2', use_fast=False)

# Noise scheduler: defines the noise removal schedule across 30 steps.
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder='scheduler')

# CLIP Text Encoders: convert text prompts into conditioning vectors.
# SDXL uses two CLIP encoders with different architecture sizes.
text_enc1 = CLIPTextModel.from_pretrained(
    MODEL_PATH, subfolder='text_encoder', torch_dtype=DTYPE)
text_enc2 = CLIPTextModelWithProjection.from_pretrained(
    MODEL_PATH, subfolder='text_encoder_2', torch_dtype=DTYPE)

# CLIP Image Encoder: encodes the garment image for ip-adapter conditioning.
img_enc = CLIPVisionModelWithProjection.from_pretrained(
    MODEL_PATH, subfolder='image_encoder', torch_dtype=DTYPE)

# VAE: compresses 768×1024 images to 96×128 latents (64× smaller).
# All 30 denoising steps run in latent space — much faster than full resolution.
vae = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder='vae', torch_dtype=DTYPE)

# Garment UNet Encoder: extracts garment features that guide the main UNet.
# ~5 GB. Runs at every denoising step to inject garment texture/shape.
unet_enc = UNet2DConditionModel_ref.from_pretrained(
    MODEL_PATH, subfolder='unet_encoder', torch_dtype=DTYPE)
unet_enc.requires_grad_(False)   # inference only, no gradient needed

# Main UNet: the denoiser. Takes noisy latents + conditioning → less noisy latents.
# ~6 GB. The largest component. Runs 30 times (once per denoising step).
unet = UNet2DConditionModel.from_pretrained(
    MODEL_PATH, subfolder='unet', torch_dtype=DTYPE)
unet.requires_grad_(False)

# Assemble the full pipeline from components.
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

# BASELINE: move everything to GPU at once.
# On a 3090 with 24 GB, this works. On an 8 GB GPU, this would fail.
pipe.to(DEVICE)
pipe.unet_encoder.to(DEVICE)

load_time = time.time() - t_load
log(f'All models on GPU in {load_time:.1f}s  |  VRAM: {vram_gb():.2f} GB')

# ── Stage 3: Preprocess Input Images ───────────────────────────────────────────
log('Preprocessing inputs...')

# Load and resize both images to the target resolution.
garm_img  = prepare_garment(GARMENT_IMG, WIDTH, HEIGHT)
human_img = Image.open(HUMAN_IMG).convert('RGB').resize((WIDTH, HEIGHT))

# OpenPose: detect body keypoints. Runs at 384×512 (faster, sufficient detail).
keypoints = openpose_model(human_img.resize((384, 512)))

# Human parsing: segment body into regions. Also at 384×512.
model_parse, _ = parsing_model(human_img.resize((384, 512)))

# Generate mask: the white region = pixels the model will replace.
# 'upper_body' means we target the torso/shirt region.
mask, mask_gray = get_mask_location('hd', 'upper_body', model_parse, keypoints)
mask = mask.resize((WIDTH, HEIGHT))

# Create a greyed-out preview of the masked area (for visualisation).
mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

# Prepare human image for DensePose (expects BGR numpy array).
human_arg = _apply_exif_orientation(human_img.resize((384, 512)))
human_arg = convert_PIL_to_numpy(human_arg, format='BGR')

log('Running DensePose (UV body map)...')
# DensePose arguments: config file + checkpoint + output type.
dp_args = apply_net.create_argument_parser().parse_args((
    'show', DENSEPOSE_CFG, DENSEPOSE_PKL,
    'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda',
))
# Run DensePose and convert BGR output to RGB.
pose_img = dp_args.func(dp_args, human_arg)[:, :, ::-1]
pose_img = Image.fromarray(pose_img).resize((WIDTH, HEIGHT))

# ── Stage 4: Run Diffusion Inference ───────────────────────────────────────────
log(f'Running inference ({STEPS} steps)...')
torch.cuda.reset_peak_memory_stats()   # Reset to measure only inference VRAM
t_inf = time.time()

with torch.no_grad():             # No gradient tracking needed during inference
    with torch.cuda.amp.autocast():  # Automatic mixed precision for stability

        # Encode text prompts into conditioning vectors.
        # Positive prompt: what we want.
        # Negative prompt: what we don't want (common defects to avoid).
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

        # Separately encode the garment description for cross-attention.
        prompt_c = f'a photo of {GARMENT_DESC}'
        with torch.inference_mode():
            (prompt_embeds_c, _, _, _) = pipe.encode_prompt(
                prompt_c,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=negative_prompt,
            )

        # Convert pose image and garment image to GPU tensors.
        pose_img_t  = tensor_transform(pose_img).unsqueeze(0).to(DEVICE, DTYPE)
        garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(DEVICE, DTYPE)

        # Fixed random seed = reproducible results across experiments.
        generator = torch.Generator(DEVICE).manual_seed(SEED)

        # Run the full pipeline. This is the 30-step denoising loop.
        # Internally: encode inputs → add noise → denoise × 30 → decode
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

# Save benchmark metrics for comparison with optimised versions.
metrics = {
    'step':             'step0_baseline',
    'gpu':              'RTX 3090 24GB',
    'mode':             'all_gpu_no_offload',
    'optimisations':    [],
    'dtype':            'float16',
    'resolution':       f'{WIDTH}x{HEIGHT}',
    'steps':            STEPS,
    'guidance_scale':   GUIDANCE,
    'seed':             SEED,
    'vram_peak_gb':     round(vram_peak, 2),
    'load_time_s':      round(load_time, 1),
    'inference_time_s': round(inf_time, 1),
    'output':           out_path,
}
with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

log('')
log('=' * 55)
log(f'  Step        : 0 — Baseline (no optimisation)')
log(f'  Output      : {out_path}')
log(f'  Resolution  : {WIDTH}x{HEIGHT}  |  Steps: {STEPS}')
log(f'  VRAM peak   : {vram_peak:.2f} GB')
log(f'  Load time   : {load_time:.1f}s')
log(f'  Infer time  : {inf_time:.1f}s')
log('=' * 55)
log('Next: run research/step1_xformers/script.py')
