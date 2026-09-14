"""
QUANT COMPARISON -- Step 0: Baseline (fp16, no optimisation)
=============================================================
All models loaded to GPU in float16. No quantisation, no xformers.
This is the colour reference -- all subsequent steps are compared against this.
"""

import sys, os, time, json
from pathlib import Path

ROOT     = str(Path(__file__).parent.parent.parent.parent.resolve())
IDMVTON  = os.path.join(ROOT, 'IDM-VTON')
DEMO_DIR = os.path.join(IDMVTON, 'gradio_demo')
sys.path.insert(0, IDMVTON)
sys.path.insert(0, DEMO_DIR)

import torch
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
def log(msg): print(f'[BASELINE] {msg}', flush=True)
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

log('Loading OpenPose...')
openpose_model = OpenPose(0)
openpose_model.preprocessor.body_estimation.model.to(DEVICE)
log('Loading Human Parsing...')
parsing_model = Parsing(0)

log('Loading models to GPU...')
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

metrics = {'step': 'baseline', 'vram_peak_gb': round(vram_peak, 2),
           'inference_time_s': round(inf_time, 1), 'optimisations': 'none',
           'weight_precision': 'float16', 'resolution': f'{WIDTH}x{HEIGHT}'}
with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

log(f'VRAM peak: {vram_peak:.2f} GB  |  Time: {inf_time:.1f}s')
log(f'Output: {OUTPUT_DIR}/tryon_output.png')
