"""
Run only the 4 new lower-body products through all 4 configs.
Self-contained — shares no top-level state with run_all.py.
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
PRODUCTS_DIR  = Path(__file__).parent.resolve()
HUMAN_IMG     = os.path.join(DEMO_DIR, 'example', 'human', '00034_00.jpg')
RESULTS_DIR   = PRODUCTS_DIR / 'results'
WIDTH, HEIGHT = 768, 1024
STEPS         = 30
GUIDANCE      = 2.0
SEED          = 42
DTYPE         = torch.float16
DEVICE        = 'cuda:0'
DENSEPOSE_CFG = os.path.join(IDMVTON, 'configs', 'densepose_rcnn_R_50_FPN_s1x.yaml')
DENSEPOSE_PKL = os.path.join(IDMVTON, 'ckpt', 'densepose', 'model_final_162be9.pkl')

PRODUCTS = [
    (
        'Capri Denim Jeans Summer Vacation dark blue.png',
        'dark blue capri denim jeans',
        'lower_body',
        'capri_jeans_dark_blue',
    ),
    (
        'Capri Denim Jeans Summer Vacation washed blue.png',
        'washed blue capri denim jeans',
        'lower_body',
        'capri_jeans_washed_blue',
    ),
    (
        'Casual Street Denim Shorts dark blue.png',
        'dark blue casual street denim shorts',
        'lower_body',
        'shorts_dark_blue',
    ),
    (
        'Casual Street Denim Short washed blue.png',
        'washed blue casual street denim shorts',
        'lower_body',
        'shorts_washed_blue',
    ),
]

CONFIGS = ['int8', 'int4', 'int4_attn_slicing', 'int8_xformers_upcast']

def log(msg): print(msg, flush=True)
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

def quantize_int8(model, min_params=2048):
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

# ── Load preprocessing models once ──────────────────────────────────────────────
log('=' * 70)
log('Loading OpenPose and Human Parsing...')
log('=' * 70)
openpose_model = OpenPose(0)
openpose_model.preprocessor.body_estimation.model.to(DEVICE)
parsing_model  = Parsing(0)

human_img_raw  = Image.open(HUMAN_IMG).convert('RGB').resize((WIDTH, HEIGHT))
keypoints      = openpose_model(human_img_raw.resize((384, 512)))
model_parse, _ = parsing_model(human_img_raw.resize((384, 512)))
human_arg = _apply_exif_orientation(human_img_raw.resize((384, 512)))
human_arg = convert_PIL_to_numpy(human_arg, format='BGR')
log('Running DensePose...')
dp_args = apply_net.create_argument_parser().parse_args((
    'show', DENSEPOSE_CFG, DENSEPOSE_PKL, 'dp_segm', '-v',
    '--opts', 'MODEL.DEVICE', 'cuda'))
pose_img = Image.fromarray(dp_args.func(dp_args, human_arg)[:, :, ::-1]).resize((WIDTH, HEIGHT))
log('Human preprocessing done.')

def load_pipeline(config_name):
    log(f'\n{"=" * 70}')
    log(f'Loading config: {config_name}')
    log(f'{"=" * 70}')
    upcast = (config_name == 'int8_xformers_upcast')
    tokenizer_one   = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder='tokenizer',   use_fast=False)
    tokenizer_two   = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder='tokenizer_2', use_fast=False)
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder='scheduler')
    text_enc1 = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder='text_encoder',   torch_dtype=DTYPE)
    text_enc2 = CLIPTextModelWithProjection.from_pretrained(MODEL_PATH, subfolder='text_encoder_2', torch_dtype=DTYPE)
    img_enc   = CLIPVisionModelWithProjection.from_pretrained(MODEL_PATH, subfolder='image_encoder', torch_dtype=DTYPE)
    vae       = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder='vae', torch_dtype=DTYPE)
    unet_enc  = UNet2DConditionModel_ref.from_pretrained(MODEL_PATH, subfolder='unet_encoder',
                                                          torch_dtype=DTYPE, upcast_attention=upcast)
    unet_enc.requires_grad_(False)
    unet      = UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder='unet',
                                                      torch_dtype=DTYPE, upcast_attention=upcast)
    unet.requires_grad_(False)
    if config_name in ('int8', 'int8_xformers_upcast'):
        n, b, a   = quantize_int8(unet)
        n2, b2, a2 = quantize_int8(unet_enc)
    else:
        n, b, a   = quantize_int4(unet)
        n2, b2, a2 = quantize_int4(unet_enc)
    layers_q     = n + n2
    weight_saved = (b + b2) - (a + a2)
    log(f'  Weights: {(b+b2)}MB -> {(a+a2)}MB (saved {weight_saved}MB)')
    pipe = TryonPipeline.from_pretrained(MODEL_PATH, unet=unet, vae=vae,
        feature_extractor=CLIPImageProcessor(), text_encoder=text_enc1,
        text_encoder_2=text_enc2, tokenizer=tokenizer_one, tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler, image_encoder=img_enc, torch_dtype=DTYPE)
    pipe.unet_encoder = unet_enc
    pipe.to(DEVICE)
    pipe.unet_encoder.to(DEVICE)
    if config_name == 'int4_attn_slicing':
        pipe.enable_attention_slicing(slice_size=1)
        try:
            pipe.unet_encoder.set_attention_slice(1)
        except Exception:
            pass
        log('  Attention slicing: enabled')
    if config_name == 'int8_xformers_upcast':
        try:
            pipe.enable_xformers_memory_efficient_attention()
            pipe.unet_encoder.enable_xformers_memory_efficient_attention()
            log('  xformers: enabled')
        except Exception as e:
            log(f'  xformers: {e}')
    return pipe, layers_q, weight_saved

def unload_pipeline(pipe):
    pipe.unet_encoder.to('cpu')
    pipe.to('cpu')
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def run_product(pipe, product, config_name, layers_q, weight_saved):
    filename, garment_desc, mask_category, slug = product
    out_dir = RESULTS_DIR / slug / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    garm_img = prepare_garment(str(PRODUCTS_DIR / filename), WIDTH, HEIGHT)
    mask, mask_gray = get_mask_location('hd', mask_category, model_parse, keypoints)
    mask = mask.resize((WIDTH, HEIGHT))
    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img_raw)
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
                cloth=garm_tensor, mask_image=mask, image=human_img_raw,
                height=HEIGHT, width=WIDTH,
                ip_adapter_image=garm_img.resize((WIDTH, HEIGHT)), guidance_scale=GUIDANCE,
            )[0]
    inf_time  = time.time() - t_inf
    vram_peak = vram_gb()
    images[0].save(out_dir / 'tryon_output.png')
    mask_gray.save(out_dir / 'mask_preview.png')
    human_img_raw.save(out_dir / 'human_input.png')
    garm_img.save(out_dir / 'garment_input.png')
    colour_accurate = config_name in ('int8', 'int4')
    metrics = {
        'product': slug, 'config': config_name, 'garment_desc': garment_desc,
        'mask_category': mask_category, 'vram_peak_gb': round(vram_peak, 2),
        'inference_time_s': round(inf_time, 1), 'layers_quantised': layers_q,
        'weight_saved_mb': weight_saved, 'colour_accurate': colour_accurate,
        'resolution': f'{WIDTH}x{HEIGHT}',
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    log(f'    [{config_name}] {slug}: {vram_peak:.2f} GB | {inf_time:.1f}s | colour_accurate={colour_accurate}')
    return metrics

# ── Main loop ────────────────────────────────────────────────────────────────────
all_metrics = []
total_start = time.time()

for config in CONFIGS:
    pipe, layers_q, weight_saved = load_pipeline(config)
    log(f'\nRunning {len(PRODUCTS)} new products through [{config}]...')
    for product in PRODUCTS:
        m = run_product(pipe, product, config, layers_q, weight_saved)
        all_metrics.append(m)
    unload_pipeline(pipe)
    log(f'Config [{config}] done. GPU cleared.')

total_time = time.time() - total_start
log('\n' + '=' * 70)
log(f'ALL DONE — {len(all_metrics)} runs in {total_time/60:.1f} min')
log(f'Results: {RESULTS_DIR}')
log('=' * 70)
