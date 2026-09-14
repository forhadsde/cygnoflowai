# Step 0 — Baseline Benchmark Report

**Method:** All models loaded directly to GPU, no memory optimisation  
**Resolution:** 768 × 1024  
**Hardware:** NVIDIA RTX 3090 (24 GB VRAM)  
**Result:** 23.2s inference, 16.29 GB VRAM peak

---

## What We Did and Why — A Complete Walkthrough

This document explains every step of the baseline experiment from scratch.
No prior AI knowledge is assumed.

---

### Part 1: What Is IDM-VTON?

IDM-VTON (Improving Diffusion Models for Virtual Try-On) is an AI system
that can dress a person in a garment they have never physically worn.

Given two images:
- A **human photo** (person standing, visible body)
- A **garment photo** (clothing item on white background)

It generates a realistic new image: the same person wearing the garment,
with correct lighting, wrinkles, and body shape.

This is commercially valuable for e-commerce — customers can see themselves
wearing clothes before buying.

**How it works (simplified):**
```
Human photo ──┐
              ├──► AI Model ──► Try-on result
Garment photo ┘
```

The AI does not just paste the garment on. It understands the body pose,
generates realistic fabric draping, and preserves the person's identity.

---

### Part 2: What Is a Diffusion Model?

IDM-VTON is built on a **diffusion model**. Here is the intuition:

Imagine you have a photo. You slowly add random noise (static) to it,
step by step, until it becomes pure static. A diffusion model learns to
reverse this process — it learns to remove noise, step by step.

```
Pure noise → [remove noise × 30 steps] → Clean image
```

At inference time:
1. Start with random noise
2. Run 30 denoising steps, each guided by the garment and human inputs
3. After step 30, you have a clean try-on image

Each denoising step runs the full neural network. This is why inference
takes ~20 seconds — 30 passes through a ~6 GB network.

---

### Part 3: The Model Components

IDM-VTON is not one model. It is a **pipeline** of 6 separate models,
each doing a specific job:

#### 3.1 OpenPose
- **What it does:** Detects the person's body pose — where the head,
  shoulders, arms, hips are
- **Why we need it:** The try-on model needs to know the body position
  to place the garment correctly
- **Output:** A skeleton diagram of joint positions

#### 3.2 Human Parsing (ONNX model)
- **What it does:** Segments the image into regions — skin, hair, top
  clothing, bottom clothing, background
- **Why we need it:** To know exactly which pixels to replace with the
  new garment (we only want to change the top clothing region)
- **Output:** A colour-coded map of body regions

#### 3.3 DensePose (Detectron2)
- **What it does:** Creates a detailed UV body map — maps each pixel of
  the person's body to a 3D body surface coordinate
- **Why we need it:** The main model uses this to understand the 3D
  geometry of the body, enabling realistic fabric draping
- **Output:** A coloured body map where colour = 3D position

#### 3.4 CLIP Text + Image Encoders
- **What it does:** Converts text descriptions ("model is wearing a shirt")
  and garment images into numerical vectors that the AI understands
- **Why we need it:** The denoising model is guided by these vectors —
  they tell it what the final image should look like
- **Output:** Float16 tensors (lists of numbers)

#### 3.5 VAE (Variational Autoencoder)
- **What it does:** Compresses images into a smaller "latent space" for
  processing, then decompresses the result back into a full image
- **Why we need it:** Running 30 denoising steps on a full 768×1024 image
  would be extremely slow. The VAE compresses it to 96×128 (64× smaller),
  and all 30 steps run in this compressed space
- **Output:** Latent tensors (compressed image representations)

#### 3.6 UNet (The Main Denoiser)
- **What it does:** The core of the diffusion model. Takes noisy latents
  and predicts how to remove the noise, guided by the pose, garment,
  and text
- **Why we need it:** This is the model that actually "generates" the image
- **Size:** ~6 GB — the largest component
- **Output:** Slightly-less-noisy latents (repeated 30 times)

#### 3.7 Garment UNet Encoder
- **What it does:** Analyses the garment image and extracts detailed
  features (texture, shape, colour patterns)
- **Why we need it:** These features are injected into the main UNet at
  each denoising step, ensuring the generated garment matches the input
- **Size:** ~5 GB — second largest component

---

### Part 4: What Is VRAM and Why Does It Matter?

**VRAM** (Video RAM) is the memory on your GPU (graphics card).

When you run an AI model, all its weights (numbers that define the model)
must be loaded into VRAM. If your model is larger than your VRAM, it
cannot run — or must use workarounds like storing parts on regular RAM
(much slower).

```
RTX 3090 VRAM: 24 GB total
  ├── IDM-VTON models: ~14 GB
  ├── Activations (intermediate computations): ~2 GB
  └── Free: ~8 GB
```

Our baseline used **16.29 GB peak VRAM** — leaving only ~7.7 GB free.
This means the pipeline barely fits on a 24 GB card, and would be
completely impossible on an 8 GB GPU without optimisation.

---

### Part 5: Environment Setup — How We Prepared the System

Before running any code, we needed to install all the required software.

**Step 5.1 — Install Miniconda**

Miniconda is a Python package manager. We use it to create an isolated
environment (`idm`) with exactly the right package versions.

```powershell
# Download and install Miniconda silently
.\Miniconda3.exe /InstallationType=JustMe /S /D=C:\Users\Rey\miniconda3
```

**Why isolated environment?** Different AI projects need different
versions of the same library. An environment keeps them separate so
they don't conflict.

**Step 5.2 — Create the Environment**

```powershell
conda create -n idm python=3.10 -y
```

We use Python 3.10 because all IDM-VTON dependencies are tested with it.

**Step 5.3 — Install PyTorch with CUDA**

```powershell
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

- **PyTorch:** The deep learning framework. All model computations run through it.
- **CUDA 12.1:** The NVIDIA GPU programming toolkit. Without it, everything
  runs on CPU (~100× slower).
- **Why version 2.1.0?** IDM-VTON was developed and tested with this version.
  Newer versions can break compatibility.

**Step 5.4 — Install Dependencies**

```powershell
pip install diffusers==0.25.0 transformers==4.36.2 accelerate==0.25.0 ...
```

Key packages:
- `diffusers`: Hugging Face library for diffusion models (the pipeline)
- `transformers`: For CLIP encoders
- `onnxruntime-gpu`: Runs the human parsing ONNX models on GPU
- `huggingface_hub==0.20.3`: Downloads model weights from HuggingFace

**Step 5.5 — Download Model Checkpoints**

```powershell
# DensePose model (~253 MB) — from HuggingFace
hf_hub_download('yisol/IDM-VTON', 'densepose/model_final_162be9.pkl')

# Human parsing ONNX models (~100 MB each)
hf_hub_download('yisol/IDM-VTON', 'humanparsing/parsing_atr.onnx')
hf_hub_download('yisol/IDM-VTON', 'humanparsing/parsing_lip.onnx')
```

The main diffusion model (~12 GB) downloads automatically on first run
from HuggingFace (`yisol/IDM-VTON`).

---

### Part 6: The Inference Script — Step by Step

Here we walk through exactly what `script.py` does, line by line in plain
English.

**Step 6.1 — Configuration**

```python
MODEL_PATH   = 'yisol/IDM-VTON'   # HuggingFace model ID
WIDTH, HEIGHT = 768, 1024          # Output resolution
STEPS        = 30                  # Number of denoising steps
GUIDANCE     = 2.0                 # How strongly to follow the prompt
SEED         = 42                  # Fixed seed for reproducibility
DTYPE        = torch.float16       # Use 16-bit floats (half the memory of 32-bit)
DEVICE       = 'cuda:0'            # Use the first GPU
```

- **float16:** By default, numbers in Python use 32 bits. float16 uses 16 bits,
  halving memory usage with minimal quality loss. This is standard for
  modern GPU inference.
- **SEED = 42:** Without a fixed seed, the random noise is different every run,
  producing different results. A fixed seed makes experiments reproducible.

**Step 6.2 — Load Preprocessing Models**

```python
openpose_model = OpenPose(0)       # Load pose detector
parsing_model  = Parsing(0)        # Load body segmenter
```

These are small models (~50-200 MB). They run first to prepare the inputs
for the main diffusion model.

**Step 6.3 — Load Diffusion Models to GPU**

```python
text_enc1 = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder='text_encoder', torch_dtype=DTYPE)
text_enc2 = CLIPTextModelWithProjection.from_pretrained(...)
img_enc   = CLIPVisionModelWithProjection.from_pretrained(...)
vae       = AutoencoderKL.from_pretrained(...)
unet_enc  = UNet2DConditionModel_ref.from_pretrained(...)  # garment encoder
unet      = UNet2DConditionModel.from_pretrained(...)      # main denoiser

pipe = TryonPipeline.from_pretrained(MODEL_PATH, unet=unet, vae=vae, ...)
pipe.to(DEVICE)             # move entire pipeline to GPU
pipe.unet_encoder.to(DEVICE)  # move garment encoder to GPU
```

All models are loaded directly into GPU VRAM. This is the **simplest**
strategy: everything is ready on the GPU, no transfers needed during inference.
This is fast but requires a lot of VRAM (~16 GB peak).

**First run only:** The 12 GB model downloads from HuggingFace before
loading. This took ~10 minutes. Subsequent runs load from local cache
in ~10-15 seconds.

**Step 6.4 — Preprocess Inputs**

```python
# Open and resize input images
garm_img  = Image.open(GARMENT_IMG).convert('RGB').resize((768, 1024))
human_img = Image.open(HUMAN_IMG).convert('RGB').resize((768, 1024))

# Detect body pose
keypoints = openpose_model(human_img.resize((384, 512)))

# Segment body regions
model_parse, _ = parsing_model(human_img.resize((384, 512)))

# Create mask: which pixels should the model replace?
mask, mask_gray = get_mask_location('hd', 'upper_body', model_parse, keypoints)
```

The mask tells the model: "only change these pixels (the upper body clothing
area) — keep everything else the same." This is why the person's face,
hands, and background are preserved in the output.

```python
# Run DensePose: create the UV body map
pose_img = dp_args.func(dp_args, human_arg)[:, :, ::-1]
```

DensePose takes the longest of the preprocessing steps because it runs
a full object detection network (Detectron2).

**Step 6.5 — Run Inference**

```python
with torch.no_grad():           # disable gradient tracking (not training, just inferring)
    with torch.cuda.amp.autocast():  # automatic precision management

        # Encode the text prompts into vectors
        prompt_embeds, negative_prompt_embeds, ... = pipe.encode_prompt(
            'model is wearing shirt',
            negative_prompt='monochrome, lowres, bad anatomy, worst quality, low quality'
        )

        # Run the full diffusion pipeline
        images = pipe(
            prompt_embeds=prompt_embeds,
            pose_img=pose_img_t,           # body pose conditioning
            text_embeds_cloth=...,         # garment text conditioning
            cloth=garm_tensor,             # garment image conditioning
            mask_image=mask,               # which region to change
            image=human_img,               # the person photo
            num_inference_steps=30,        # 30 denoising steps
            guidance_scale=2.0,
            ip_adapter_image=garm_img,     # garment for ip-adapter
        )[0]
```

The pipeline internally:
1. Encodes human + garment images through the VAE into latent space
2. Adds random noise to create the starting point
3. Runs 30 denoising steps (each step: garment UNet → main UNet → scheduler)
4. Decodes the final latent back to a full image through the VAE

**Step 6.6 — Save Results**

```python
images[0].save('results/tryon_output.png')

metrics = {
    'vram_peak_gb':     16.29,
    'inference_time_s': 23.2,
    'load_time_s':      1255.9,   # only slow on first run (download)
    ...
}
```

---

### Part 7: Results

| Metric | Value |
|---|---|
| Output resolution | 768 × 1024 px |
| Inference time | **23.2 seconds** |
| Model load time | 10–15s (after first-run download) |
| Peak VRAM used | **16.29 GB** |
| VRAM free (of 24 GB) | ~7.7 GB |
| Inference steps | 30 |
| Precision | float16 |

**Output images:**

- `results/tryon_output.png` — the generated try-on result
- `results/human_input.png` — the input person photo
- `results/garment_input.png` — the input garment photo
- `results/mask_preview.png` — the region mask (what the model changed)

---

### Part 8: What Did We Learn? What Are the Bottlenecks?

**Memory bottleneck — the attention layers**

The UNet uses **self-attention** at multiple scales. Attention computes
relationships between every pair of image patches, which scales as O(n²):
double the resolution → 4× the attention memory. At 768×1024, this alone
uses ~4-5 GB of the 16.29 GB peak.

**Memory bottleneck — model weight storage**

Just storing the model weights takes ~13 GB. This is fixed regardless of
resolution. The only way to reduce it is compression (quantisation).

**Speed bottleneck — 30 serial denoising steps**

Each step requires a full forward pass through two UNets (~11 GB of
models). These steps cannot be parallelised — each step depends on the
output of the previous one.

**Conclusion:** The baseline is fast (23.2s) but memory-hungry (16.29 GB).
The next step applies **memory-efficient attention** (xformers) to directly
attack the attention memory bottleneck without sacrificing quality.

---

### Part 9: Why This Matters for the PhD

The Edinburgh PhD asks: *how do we automatically map models onto efficient
spatial systems?* This baseline experiment manually maps one model onto one
system and measures the result.

The research question this raises:
- Can we profile ANY model and automatically identify the same bottlenecks?
- Can a compiler pass automatically replace standard attention with
  memory-efficient attention?
- Can quantisation be applied automatically at the graph level without
  manual code changes?

The baseline gives us the ground truth numbers. Every subsequent experiment
is a data point answering these questions.

---

*Next: [Step 1 — xformers Memory-Efficient Attention](../step1_xformers/REPORT.md)*
