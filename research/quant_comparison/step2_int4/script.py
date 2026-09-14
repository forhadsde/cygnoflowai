"""
QUANT COMPARISON -- Step 2: INT4 Weight Quantisation (W4A16)
=============================================================
Weights compressed from float16 (2 bytes) to int4 (0.5 bytes) via bit packing.
No xformers -- preserves colour accuracy vs baseline.
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
def log(msg): print(f'[INT4] {msg}', flush=True)
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

class QuantizedLinearInt4(nn.Module):
    def __init__(self, in_features, out_features, weight_fp16, bias=None):
        super().__init__()
        self.in_features = in_features
        w = weight_fp16.float()
        scale = w.abs().max(dim=1, keepdim=True).values / 7.0
        scale = scale.clamp(min=1e-8)
        w_q = (w / scale).round().clamp(-7, 7).to(torch.int8)
        w_u = w_q.to(torch.int16) + 7
        if in_features % 2 != 0:
            w_u = torch.cat([w_u, torch.zeros(out_features, 1, dtype=torch.int16)], dim=1)
        low  = w_u[:, 0::2]
        high = w_u[:, 1::2]
        packed = (low | (high << 4)).to(torch.uint8).view(torch.int8)
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
        w_q = w_u[:, :self.in_features] - 7
        w_fp = w_q.to(x.dtype) * self.scale.unsqueeze(1)
        return nn.functional.linear(x, w_fp, self.bias)

def quantize_to_int4(model, min_params=2048):
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

log('Loading OpenPose...')
openpose_model = OpenPose(0)
openpose_model.preprocessor.body_estimation.model.to(DEVICE)
log('Loading Human Parsing...')
parsing_model = Parsing(0)

log('Loading models to CPU for quantisation...')
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

log('Quantising UNet to INT4...')
n, before, after = quantize_to_int4(unet)
log(f'  UNet: {n} layers | {before} MB -> {after} MB (saved {before-after} MB)')
n2, before2, after2 = quantize_to_int4(unet_enc)
log(f'  UNet encoder: {n2} layers | {before2} MB -> {after2} MB (saved {before2-after2} MB)')
log(f'  Total saved: {(before+before2)-(after+after2)} MB')

pipe = TryonPipeline.from_pretrained(MODEL_PATH, unet=unet, vae=vae,
    feature_extractor=CLIPImageProcessor(), text_encoder=text_enc1,
    text_encoder_2=text_enc2, tokenizer=tokenizer_one, tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler, image_encoder=img_enc, torch_dtype=DTYPE)
pipe.unet_encoder = unet_enc
pipe.to(DEVICE)
pipe.unet_encoder.to(DEVICE)

load_time = time.time() - t_load
log(f'Ready in {load_time:.1f}s  |  VRAM: {vram_gb():.2f} GB')

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

log(f'Running inference ({STEPS} steps)...')
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

images[0].save(os.path.join(OUTPUT_DIR, 'tryon_output.png'))
mask_gray.save(os.path.join(OUTPUT_DIR, 'mask_preview.png'))
human_img.save(os.path.join(OUTPUT_DIR, 'human_input.png'))
garm_img.save(os.path.join(OUTPUT_DIR, 'garment_input.png'))

metrics = {'step': 'int4', 'vram_peak_gb': round(vram_peak, 2),
           'inference_time_s': round(inf_time, 1), 'optimisations': 'int4_weights',
           'weight_precision': 'int4 weights (bit-packed), float16 activations',
           'layers_quantised': n + n2, 'weight_saved_mb': (before+before2)-(after+after2),
           'resolution': f'{WIDTH}x{HEIGHT}'}
with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

log(f'VRAM peak: {vram_peak:.2f} GB  |  Time: {inf_time:.1f}s')
log(f'Output: {OUTPUT_DIR}/tryon_output.png')
