"""
IDM-VTON RTX 3090 Inference — All models on GPU (24 GB VRAM)
-------------------------------------------------------------
No CPU offloading. Expected: ~30-40s vs ~160s on 8 GB.

Run from repo root:
    conda run -n idm python tryon_3090.py
"""

import sys, os, time, json, gc
from pathlib import Path

ROOT     = str(Path(__file__).parent.resolve())
IDMVTON  = os.path.join(ROOT, 'IDM-VTON')
DEMO_DIR = os.path.join(IDMVTON, 'gradio_demo')

# Order matters: IDM-VTON first so `from src.*` resolves there;
# gradio_demo second so apply_net / densepose / detectron2 resolve there.
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
MODEL_PATH   = 'yisol/IDM-VTON'          # downloads to HF cache on first run
GARMENT_IMG  = os.path.join(DEMO_DIR, 'example', 'cloth', '04469_00.jpg')
HUMAN_IMG    = os.path.join(DEMO_DIR, 'example', 'human', '00034_00.jpg')
OUTPUT_DIR   = os.path.join(ROOT, 'results', '3090')
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

def log(msg): print(f'[3090] {msg}', flush=True)

def vram_gb():
    return torch.cuda.max_memory_allocated() / 1024**3

tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# ── Verify prerequisites ────────────────────────────────────────────────────────
if not os.path.exists(DENSEPOSE_PKL):
    raise FileNotFoundError(
        f"DensePose checkpoint missing:\n  {DENSEPOSE_PKL}\n"
        "Run setup_idm_windows.ps1 or download it manually."
    )
for onnx in ('parsing_atr.onnx', 'parsing_lip.onnx'):
    p = os.path.join(IDMVTON, 'ckpt', 'humanparsing', onnx)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Human parsing model missing:\n  {p}\n"
            "Run setup_idm_windows.ps1 or download it manually."
        )

# ── Load preprocessing models ──────────────────────────────────────────────────
log('Loading OpenPose...')
openpose_model = OpenPose(0)
openpose_model.preprocessor.body_estimation.model.to(DEVICE)

log('Loading human parsing (ONNX)...')
parsing_model = Parsing(0)

# ── Load diffusion models — all to GPU from the start ─────────────────────────
log('Loading IDM-VTON models to GPU...')
torch.cuda.reset_peak_memory_stats()
t_load = time.time()

tokenizer_one  = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder='tokenizer',   use_fast=False)
tokenizer_two  = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder='tokenizer_2', use_fast=False)
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder='scheduler')

text_enc1 = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder='text_encoder',   torch_dtype=DTYPE)
text_enc2 = CLIPTextModelWithProjection.from_pretrained(MODEL_PATH, subfolder='text_encoder_2', torch_dtype=DTYPE)
img_enc   = CLIPVisionModelWithProjection.from_pretrained(MODEL_PATH, subfolder='image_encoder', torch_dtype=DTYPE)
vae       = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder='vae', torch_dtype=DTYPE)

unet_enc  = UNet2DConditionModel_ref.from_pretrained(MODEL_PATH, subfolder='unet_encoder', torch_dtype=DTYPE)
unet_enc.requires_grad_(False)

unet = UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder='unet', torch_dtype=DTYPE)
unet.requires_grad_(False)

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

# Put everything on GPU — 3090 has 24 GB, all models total ~14 GB fp16.
pipe.to(DEVICE)
pipe.unet_encoder.to(DEVICE)

load_time = time.time() - t_load
log(f'All models on GPU in {load_time:.1f}s  |  VRAM: {vram_gb():.2f} GB')

# ── Prepare input images ───────────────────────────────────────────────────────
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

# ── Inference ──────────────────────────────────────────────────────────────────
log(f'Running inference ({STEPS} steps)...')
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

# ── Save outputs ───────────────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, 'tryon_output.png')
images[0].save(out_path)
mask_gray.save(os.path.join(OUTPUT_DIR, 'mask_preview.png'))
human_img.save(os.path.join(OUTPUT_DIR, 'human_input.png'))
garm_img.save(os.path.join(OUTPUT_DIR, 'garment_input.png'))

metrics = {
    'gpu':             'RTX 3090 24GB',
    'mode':            'all_gpu_no_offload',
    'dtype':           'float16',
    'resolution':      f'{WIDTH}x{HEIGHT}',
    'steps':           STEPS,
    'guidance_scale':  GUIDANCE,
    'vram_peak_gb':    round(vram_peak, 2),
    'load_time_s':     round(load_time, 1),
    'inference_time_s': round(inf_time, 1),
    'output':          out_path,
}
with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

log('')
log('=' * 55)
log(f'  Output      : {out_path}')
log(f'  Resolution  : {WIDTH}x{HEIGHT}  |  Steps: {STEPS}')
log(f'  VRAM peak   : {vram_peak:.2f} GB')
log(f'  Load time   : {load_time:.1f}s')
log(f'  Infer time  : {inf_time:.1f}s')
log(f'  Speedup vs 8GB baseline: ~{round(160 / inf_time, 1)}x faster')
log('=' * 55)
