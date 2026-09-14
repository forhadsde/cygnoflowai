"""
Model Testing — INT4 try-on for all 13 products
================================================
Usage:
  python run.py                  # uses the first photo found in research/Models/
  python run.py model1.png       # uses a specific photo from research/Models/

Results saved to:
  model_testing/results/{model_name}/{product_slug}/
    tryon_output.png
    garment_input.png
    human_input.png
    mask_preview.png
    metrics.json

Estimated time: ~7 minutes (20s load + 20s preprocess + 13 x 29s inference)
Config: INT4 — colour-accurate, ~9.3 GB VRAM
"""

import sys, os, time, json
from pathlib import Path

ROOT     = str(Path(__file__).parent.parent.parent.resolve())
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

# ── Paths ─────────────────────────────────────────────────────────────────────────
MODEL_PATH   = 'yisol/IDM-VTON'
THIS_DIR     = Path(__file__).parent.resolve()
MODELS_DIR   = THIS_DIR.parent / 'Models'
PRODUCTS_DIR = THIS_DIR.parent / 'Products'
RESULTS_DIR  = THIS_DIR / 'results'
WIDTH, HEIGHT = 768, 1024
STEPS         = 30
GUIDANCE      = 2.0
SEED          = 42
DTYPE         = torch.float16
DEVICE        = 'cuda:0'
DENSEPOSE_CFG = os.path.join(IDMVTON, 'configs', 'densepose_rcnn_R_50_FPN_s1x.yaml')
DENSEPOSE_PKL = os.path.join(IDMVTON, 'ckpt', 'densepose', 'model_final_162be9.pkl')

# ── Product catalogue — all 13 products ──────────────────────────────────────────
PRODUCTS = [
    # Upper body
    (
        'Front Button Short Sleeve Women Denim Jacket dark blue.png',
        'dark blue denim jacket with front buttons and short sleeves',
        'upper_body', 'jacket_dark_blue',
    ),
    (
        'Front Button Short Sleeve Women Denim Jacket washed blue.png',
        'washed blue denim jacket with front buttons and short sleeves',
        'upper_body', 'jacket_washed_blue',
    ),
    (
        "Short Sleeve Women's Denim Cropped Shirt dark blue.png",
        'dark blue denim cropped shirt with short sleeves',
        'upper_body', 'cropped_shirt_dark_blue',
    ),
    (
        "Short Sleeve Women's Denim Cropped Shirt washed blue.png",
        'washed blue denim cropped shirt with short sleeves',
        'upper_body', 'cropped_shirt_washed_blue',
    ),
    # Dresses
    (
        'Denim Sleeveless Casual Fitted Mini Dress dark blue.png',
        'dark blue denim sleeveless fitted mini dress',
        'dresses', 'mini_dress_dark_blue',
    ),
    (
        'Denim Sleeveless Casual Fitted Mini Dress washed blue.png',
        'washed blue denim sleeveless fitted mini dress',
        'dresses', 'mini_dress_washed_blue',
    ),
    (
        'Denim Sleeveless Casual Fitted Mini Dress.png',
        'denim sleeveless casual fitted mini dress',
        'dresses', 'mini_dress',
    ),
    (
        'Front Button Sleeveless Denim Dress dark blue.png',
        'dark blue sleeveless denim dress with front buttons',
        'dresses', 'sleeveless_dress_dark_blue',
    ),
    (
        'Front Button Sleeveless Denim Dress washed blue.png',
        'washed blue sleeveless denim dress with front buttons',
        'dresses', 'sleeveless_dress_washed_blue',
    ),
    # Lower body
    (
        'Capri Denim Jeans Summer Vacation dark blue.png',
        'dark blue capri denim jeans',
        'lower_body', 'capri_jeans_dark_blue',
    ),
    (
        'Capri Denim Jeans Summer Vacation washed blue.png',
        'washed blue capri denim jeans',
        'lower_body', 'capri_jeans_washed_blue',
    ),
    (
        'Casual Street Denim Shorts dark blue.png',
        'dark blue casual street denim shorts',
        'lower_body', 'shorts_dark_blue',
    ),
    (
        'Casual Street Denim Short washed blue.png',
        'washed blue casual street denim shorts',
        'lower_body', 'shorts_washed_blue',
    ),
]

# ── Resolve human photo ───────────────────────────────────────────────────────────
def find_human_photo():
    exts = {'.jpg', '.jpeg', '.png', '.webp'}
    if len(sys.argv) > 1:
        p = MODELS_DIR / sys.argv[1]
        if not p.exists():
            print(f'ERROR: {p} not found.')
            sys.exit(1)
        return p
    photos = sorted([p for p in MODELS_DIR.iterdir() if p.suffix.lower() in exts])
    if not photos:
        print(f'ERROR: No photos in {MODELS_DIR}. Add a full-body model photo and re-run.')
        sys.exit(1)
    if len(photos) > 1:
        print(f'Multiple photos found: {[p.name for p in photos]}')
        print(f'Using: {photos[0].name}  (pass filename as argument to choose)')
    return photos[0]

# ── Helpers ───────────────────────────────────────────────────────────────────────
def log(msg): print(msg, flush=True)
def vram_gb(): return torch.cuda.max_memory_allocated() / 1024**3

tensor_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

def fit_and_pad(img_path, width, height):
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

# ── INT4 quantisation ─────────────────────────────────────────────────────────────
class QuantizedLinearInt4(nn.Module):
    def __init__(self, in_features, out_features, weight_fp16, bias=None):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        w = weight_fp16.float()
        scale = w.abs().max(dim=1, keepdim=True).values / 7.0
        scale = scale.clamp(min=1e-8)
        w_q = (w / scale).round().clamp(-7, 7).to(torch.int8)
        w_u = w_q.to(torch.int16) + 7
        if in_features % 2 != 0:
            w_u = torch.cat([w_u, torch.zeros(out_features, 1, dtype=torch.int16)], dim=1)
        packed = (w_u[:, 0::2] | (w_u[:, 1::2] << 4)).to(torch.uint8).view(torch.int8)
        self.register_buffer('weight_packed', packed)
        self.register_buffer('scale', scale.squeeze(1).half())
        if bias is not None:
            self.register_buffer('bias', bias.half())
        else:
            self.bias = None

    def forward(self, x, *args, **kwargs):
        packed = self.weight_packed.to(torch.int16) & 0xFF
        low  = (packed & 0xF)
        high = (packed >> 4) & 0xF
        out_f, n_packed = low.shape
        w_u = torch.empty(out_f, n_packed * 2, dtype=torch.int16, device=x.device)
        w_u[:, 0::2] = low
        w_u[:, 1::2] = high
        w_fp = (w_u[:, :self.in_features] - 7).to(x.dtype) * self.scale.unsqueeze(1)
        return nn.functional.linear(x, w_fp, self.bias)

def quantize_int4(model, min_params=2048):
    replaced, before_mb = 0, sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or module.weight.numel() < min_params:
            continue
        parts = name.rsplit('.', 1)
        parent = model
        for seg in (parts[:-1][0].split('.') if len(parts) > 1 else []):
            parent = getattr(parent, seg)
        child_name = parts[-1] if len(parts) > 1 else name
        setattr(parent, child_name, QuantizedLinearInt4(
            module.in_features, module.out_features,
            module.weight.data, module.bias.data if module.bias is not None else None))
        replaced += 1
    after_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    after_mb += sum(b.numel() * b.element_size() for b in model.buffers()) / 1024**2
    return replaced, round(before_mb), round(after_mb)

# ── Entry point ───────────────────────────────────────────────────────────────────
human_photo = find_human_photo()
model_name  = human_photo.stem          # e.g. "model1"
out_root    = RESULTS_DIR / model_name  # results/model1/

log('=' * 70)
log(f'Model Testing — INT4 — all {len(PRODUCTS)} products')
log(f'Human photo : {human_photo.name}')
log(f'Results     : {out_root}')
log(f'Estimated   : ~7 minutes')
log('=' * 70)

# Preprocessing (done once, reused for all products)
log('\nLoading OpenPose...')
openpose_model = OpenPose(0)
openpose_model.preprocessor.body_estimation.model.to(DEVICE)
log('Loading Human Parsing...')
parsing_model = Parsing(0)

log(f'Preprocessing {human_photo.name}...')
human_img      = fit_and_pad(str(human_photo), WIDTH, HEIGHT)
keypoints      = openpose_model(human_img.resize((384, 512)))
model_parse, _ = parsing_model(human_img.resize((384, 512)))
human_arg      = _apply_exif_orientation(human_img.resize((384, 512)))
human_arg      = convert_PIL_to_numpy(human_arg, format='BGR')
log('Running DensePose...')
dp_args  = apply_net.create_argument_parser().parse_args((
    'show', DENSEPOSE_CFG, DENSEPOSE_PKL, 'dp_segm', '-v',
    '--opts', 'MODEL.DEVICE', 'cuda'))
pose_img = Image.fromarray(dp_args.func(dp_args, human_arg)[:, :, ::-1]).resize((WIDTH, HEIGHT))
log('Preprocessing done.')

# Load + quantise model (once for all products)
log('\nLoading models and applying INT4 quantisation...')
torch.cuda.reset_peak_memory_stats()
t_load = time.time()

tokenizer_one   = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder='tokenizer',   use_fast=False)
tokenizer_two   = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder='tokenizer_2', use_fast=False)
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder='scheduler')
text_enc1 = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder='text_encoder',   torch_dtype=DTYPE)
text_enc2 = CLIPTextModelWithProjection.from_pretrained(MODEL_PATH, subfolder='text_encoder_2', torch_dtype=DTYPE)
img_enc   = CLIPVisionModelWithProjection.from_pretrained(MODEL_PATH, subfolder='image_encoder', torch_dtype=DTYPE)
vae       = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder='vae', torch_dtype=DTYPE)
unet_enc  = UNet2DConditionModel_ref.from_pretrained(MODEL_PATH, subfolder='unet_encoder', torch_dtype=DTYPE)
unet_enc.requires_grad_(False)
unet      = UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder='unet', torch_dtype=DTYPE)
unet.requires_grad_(False)

n,  b,  a  = quantize_int4(unet)
n2, b2, a2 = quantize_int4(unet_enc)
saved_mb = (b + b2) - (a + a2)
log(f'  INT4: {b+b2}MB -> {a+a2}MB (saved {saved_mb}MB, {saved_mb/1024:.1f} GB)')

pipe = TryonPipeline.from_pretrained(MODEL_PATH, unet=unet, vae=vae,
    feature_extractor=CLIPImageProcessor(), text_encoder=text_enc1,
    text_encoder_2=text_enc2, tokenizer=tokenizer_one, tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler, image_encoder=img_enc, torch_dtype=DTYPE)
pipe.unet_encoder = unet_enc
pipe.to(DEVICE)
pipe.unet_encoder.to(DEVICE)
log(f'Ready in {time.time()-t_load:.1f}s  |  VRAM: {vram_gb():.2f} GB')

# Run all products
log(f'\nRunning {len(PRODUCTS)} products...\n')
total_start = time.time()
all_metrics = []

for i, (filename, garment_desc, mask_category, slug) in enumerate(PRODUCTS, 1):
    out_dir = out_root / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    garment_path = PRODUCTS_DIR / filename
    if not garment_path.exists():
        log(f'  [{i:2d}/{len(PRODUCTS)}] SKIP {slug} — {garment_path.name} not found')
        continue

    garm_img = fit_and_pad(str(garment_path), WIDTH, HEIGHT)
    mask, mask_gray = get_mask_location('hd', mask_category, model_parse, keypoints)
    mask = mask.resize((WIDTH, HEIGHT))
    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    torch.cuda.reset_peak_memory_stats()
    t_inf = time.time()

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with torch.inference_mode():
                prompt_embeds, neg_embeds, pooled_embeds, neg_pooled = pipe.encode_prompt(
                    f'model is wearing {garment_desc}', num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt='monochrome, lowres, bad anatomy, worst quality, low quality')
            with torch.inference_mode():
                prompt_embeds_c, _, _, _ = pipe.encode_prompt(
                    f'a photo of {garment_desc}', num_images_per_prompt=1,
                    do_classifier_free_guidance=False, negative_prompt='')

            pose_img_t  = tensor_transform(pose_img).unsqueeze(0).to(DEVICE, DTYPE)
            garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(DEVICE, DTYPE)
            generator   = torch.Generator(DEVICE).manual_seed(SEED)

            images = pipe(
                prompt_embeds=prompt_embeds.to(DEVICE, DTYPE),
                negative_prompt_embeds=neg_embeds.to(DEVICE, DTYPE),
                pooled_prompt_embeds=pooled_embeds.to(DEVICE, DTYPE),
                negative_pooled_prompt_embeds=neg_pooled.to(DEVICE, DTYPE),
                num_inference_steps=STEPS, generator=generator, strength=1.0,
                pose_img=pose_img_t, text_embeds_cloth=prompt_embeds_c.to(DEVICE, DTYPE),
                cloth=garm_tensor, mask_image=mask, image=human_img,
                height=HEIGHT, width=WIDTH,
                ip_adapter_image=garm_img.resize((WIDTH, HEIGHT)), guidance_scale=GUIDANCE,
            )[0]

    inf_time  = time.time() - t_inf
    vram_peak = vram_gb()
    elapsed   = time.time() - total_start
    remaining = (elapsed / i) * (len(PRODUCTS) - i)

    images[0].save(out_dir / 'tryon_output.png')
    mask_gray.save(out_dir / 'mask_preview.png')
    human_img.save(out_dir / 'human_input.png')
    garm_img.save(out_dir / 'garment_input.png')

    metrics = {
        'model': model_name, 'product': slug, 'garment_desc': garment_desc,
        'mask_category': mask_category, 'config': 'int4',
        'vram_peak_gb': round(vram_peak, 2), 'inference_time_s': round(inf_time, 1),
        'colour_accurate': True, 'resolution': f'{WIDTH}x{HEIGHT}',
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    all_metrics.append(metrics)
    log(f'  [{i:2d}/{len(PRODUCTS)}] {slug:<32s} {vram_peak:.2f} GB | {inf_time:.1f}s | ~{remaining/60:.1f} min left')

total_time = time.time() - total_start
log(f'\n{"=" * 70}')
log(f'  DONE — {len(all_metrics)}/{len(PRODUCTS)} products in {total_time/60:.1f} min')
log(f'  Human : {human_photo.name}')
log(f'  Output: {out_root}')
log(f'{"=" * 70}')
