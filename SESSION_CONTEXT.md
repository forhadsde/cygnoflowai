# Cygnoflow — Session Context
**Last updated:** 2026-05-25  
**Branch:** main (clean, all pushed)  
**Last commit:** `dd2ca79` — Add full PhD research report  
**Repo:** https://github.com/forhadsde/cygnoflow.git  
**Environment:** `C:\Users\Rey\miniconda3\envs\idm\python.exe` | Windows 11 | CUDA 12.1 | RTX 3090 (24 GB)

---

## Who and Why

**Person:** Md Forhadul Islam (forhad.sde@gmail.com)  
**Project:** Cygnoflow — AI virtual try-on for a denim fashion e-commerce website  
**Dual purpose:**
1. Production feature: customer clicks "Try it on with AI" on a product page, uploads photo, gets photorealistic try-on result
2. PhD application: University of Edinburgh — "Memory Optimisation for Distributed ML Systems"

---

## What Was Built

### Research Phase (COMPLETE)

Systematic investigation of weight quantisation and attention optimisation on IDM-VTON. Five experiments:

| Experiment | Config | VRAM | Time | Colour | Verdict |
|---|---|---|---|---|---|
| Step 0 | Baseline float16 | 16.28 GB | 22.9s | ✓ Reference | Needs 24 GB GPU |
| Step 1 | INT8 (W8A16) | 11.70 GB | 25.4s | ✓ Accurate | **Production: 12 GB GPU** |
| Step 2 | INT4 (W4A16) | 9.30 GB | 28.5s | ✓ Accurate | **Production: 10 GB GPU** |
| Step 3 | INT4 + attn slicing | 9.06 GB | 42.0s | ✗ Shifted | Reject — colour drift + 48% slower |
| Step 4 | INT8 + xformers + upcast | 11.30 GB | 23.6s | ✗ Shifted | Reject — upcast is no-op inside xformers CUDA kernel |

**Key finding:** Colour drift is caused by float16 non-associativity when operation ORDER changes (xformers, attention slicing). Weight quantisation does NOT change operation order → INT8 and INT4 are colour-safe. CPU offload was considered and explicitly rejected — PCIe bandwidth makes it 2–15× slower, unacceptable for production.

**DO NOT use bitsandbytes** — destroys the torch CUDA installation on Windows (replaces cu121 with CPU-only build).

### Tooling Built (COMPLETE)

| File | What it does |
|---|---|
| `research/quant_comparison/step{0-4}/script.py` | Individual experiment scripts (do not touch) |
| `research/quant_comparison/COMPARISON.md` | Results table + analysis |
| `research/Products/run_all.py` | 13 products × 4 configs batch runner |
| `research/Products/run_new.py` | New products only (self-contained, do not import run_all) |
| `research/model_testing/run.py` | Any model photo → all 13 products, INT4, ~7 min |
| `RESEARCH_REPORT.md` | Full PhD research document (17 sections, 969 lines) |

### Product Catalogue (13 items in `research/Products/`)

**Upper body** (`upper_body` mask):
- `jacket_dark_blue`, `jacket_washed_blue`
- `cropped_shirt_dark_blue`, `cropped_shirt_washed_blue`

**Dresses** (`dresses` mask):
- `mini_dress_dark_blue`, `mini_dress_washed_blue`, `mini_dress`
- `sleeveless_dress_dark_blue`, `sleeveless_dress_washed_blue`

**Lower body** (`lower_body` mask):
- `capri_jeans_dark_blue`, `capri_jeans_washed_blue`
- `shorts_dark_blue`, `shorts_washed_blue`

### Model Photos

`research/Models/model1.png` — processed, results in `research/model_testing/results/model1/`

To add a new model photo: drop into `research/Models/`, then:
```bash
C:\Users\Rey\miniconda3\envs\idm\python.exe research/model_testing/run.py model2.png
# Results: research/model_testing/results/model2/{slug}/tryon_output.png
# Time: ~7 minutes, 9.30 GB VRAM
```

To add new products:
1. Drop PNG into `research/Products/`
2. Add entry to PRODUCTS list in `run_all.py` AND `run_new.py` (they are independent — run_new.py must NOT import run_all.py, run_all has top-level executable code)
3. Run `run_new.py` for just the new products

---

## What Is NOT Done — Next Phase (Production System)

This is the agreed next set of work. Build in this order:

### Phase 1 — Engine class (2–3 hrs)
Refactor existing script code into a reusable service class. New files:
```
engine/
  __init__.py
  pipeline.py      ← VTONEngine singleton: load() once, run(human_pil, product_id) → PIL
  preprocess.py    ← fit_and_pad, OpenPose, DensePose, Human Parsing
  registry.py      ← loads products.yaml
products.yaml      ← single source of truth for all 13 products + categories
```
Key constraint: `VTONEngine` must load the model ONCE at startup and keep it in GPU memory, not reload per request.

### Phase 2 — FastAPI REST API (2 hrs)
```
api/
  main.py          ← 4 routes (see below)
  schemas.py       ← Pydantic models
```
Endpoints:
- `GET  /api/products` → list all products
- `POST /api/tryon` → multipart (photo + product_id) → returns `{job_id, status, position_in_queue}`
- `GET  /api/tryon/{job_id}` → returns `{status, result_url, estimated_wait_seconds}`
- `GET  /results/{filename}` → serve result image

Install: `pip install fastapi uvicorn python-multipart`

### Phase 3 — Job Queue (2 hrs)
```
worker/
  celery_app.py    ← Celery(broker='redis://localhost:6379/0'), concurrency=1
  tasks.py         ← @app.task run_tryon(job_id, human_path, product_id)
```
- `concurrency=1` is mandatory — GPU cannot run two diffusion jobs simultaneously
- API process: CPU only (enqueues jobs)
- Worker process: holds VTONEngine in GPU memory, processes one job at a time
- Queue gives position + estimated wait to polling browser

Install: `pip install celery redis flower`  
Redis on Windows: `winget install Redis.Redis` or use Docker

Run worker: `celery -A worker.celery_app worker --concurrency=1 --loglevel=info`  
Monitoring: `celery -A worker.celery_app flower` → http://localhost:5555

### Phase 4 — Docker (2–3 hrs)
```
docker/
  Dockerfile
  docker-compose.yml
```
Services: `redis`, `api`, `worker` (GPU), `flower`  
Requires: NVIDIA Container Toolkit installed on host  
GPU passthrough in compose:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - capabilities: [gpu]
```

### Phase 5 — Website JS Widget (2 hrs)
Drop-in script on any product page:
```html
<button data-tryon-product="jacket_dark_blue" class="tryon-btn">Try it on with AI</button>
<script src="/static/tryon-widget.js"></script>
```
Widget handles: file/camera input → POST to /api/tryon → poll every 3s → show result side-by-side with product image. No framework, vanilla JS.

---

## Cloud Migration (after Phase 5, zero code changes)

| Component | Local | Cloud swap |
|---|---|---|
| Redis | Docker | AWS Elasticache |
| Results storage | Local volume | AWS S3 (one env var) |
| Worker GPU | RTX 3090 | EC2 g4dn.xlarge (T4 16 GB, INT8, ~$0.50/hr) |
| API | Port 8000 | Behind ALB |
| Model weights | Local path | EFS mount |

**Target cloud GPU:** T4 (16 GB) running INT8 — $0.50/hr, ~25s latency. Do NOT use CPU offload (2–15× slower, production-unacceptable).

---

## Technical Reference

### INT4 Quantisation (used in all production tooling)
- Per-channel symmetric, range −7 to +7, nibble-packed (2 weights per byte)
- `scale[i] = max(|W[i,:]|) / 7.0`
- Packing: `packed = low_nibble | (high_nibble << 4)`
- Unpacking at inference: bit-shift + subtract 7 + scale → float16
- Applied to: both UNets, all `nn.Linear` layers with ≥ 2048 parameters
- Colour: fully accurate (does not change attention computation order)

### INT8 Quantisation
- Per-channel symmetric, range −127 to +127
- `scale[i] = max(|W[i,:]|) / 127.0`
- Dequantise immediately before matmul
- Colour: fully accurate

### fit_and_pad (used for both human photos and garment images)
```python
def fit_and_pad(img_path, width=768, height=1024):
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
```

### Human Preprocessing Pipeline (done ONCE per model photo, reused across all products)
1. OpenPose → keypoints (run at 384×512)
2. Human Parsing (SCHP) → segment map (run at 384×512)
3. DensePose (Detectron2) → UV surface (run at 384×512, output resized to 768×1024)

### Key Paths
```
IDM-VTON source:    cygnoflow/IDM-VTON/
Model weights:      yisol/IDM-VTON (HuggingFace, cached locally)
DensePose config:   IDM-VTON/configs/densepose_rcnn_R_50_FPN_s1x.yaml
DensePose weights:  IDM-VTON/ckpt/densepose/model_final_162be9.pkl
Products:           research/Products/
Models:             research/Models/
Results (batch):    research/Products/results/{slug}/{config}/
Results (model):    research/model_testing/results/{model_name}/{slug}/
```

### Inference Parameters (fixed across all experiments)
- Resolution: 768×1024
- Steps: 30 (DDPM scheduler)
- Guidance scale: 2.0
- Seed: 42
- dtype: float16

---

## Commits (main branch)

| Hash | Description |
|---|---|
| `dd2ca79` | Add full PhD research report (RESEARCH_REPORT.md) |
| `5f51596` | Add model_testing runner and model1 results (13 products, INT4) |
| `afb4656` | Add 4 lower-body products and their 16 try-on results |
| `9f99671` | Add product batch runner and all 36 try-on results (9 products × 4 configs) |
| `635a8aa` | Add step4 upcast_attention experiment (negative result) |
| `32f0df0` | Add step3 INT4+attn_slicing trial |
| `ac62ba3` | Add IDM-VTON source, benchmark outputs, and .gitignore |

---

## PhD Application Notes

- **University:** University of Edinburgh
- **Programme:** Memory Optimisation for Distributed ML Systems
- **Core contribution:** Empirical characterisation of the float16 non-associativity constraint in colour-critical diffusion inference; implementation of W8A16 and W4A16 quantisation from scratch without external libraries; isolation of the CUDA kernel boundary problem that prevents upcast_attention from fixing xformers colour drift
- **Open problem identified:** FlashAttention with float32 accumulation (no public CUDA kernel exists; would fix xformers colour drift while retaining memory savings)
- **Full write-up:** `RESEARCH_REPORT.md` in repo root
