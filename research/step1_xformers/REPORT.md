# Step 1 — xformers Memory-Efficient Attention + VAE Tiling

**Method:** Replace O(n²) attention with O(n) xformers attention. Enable VAE tiling.
**Resolution:** 768 × 1024 (same as baseline — only one variable changed)
**Hardware:** NVIDIA RTX 3090 (24 GB VRAM)

---

## Results vs Baseline

| Metric | Step 0 Baseline | Step 1 xformers | Change |
|---|---|---|---|
| VRAM peak | 16.29 GB | **14.07 GB** | **-2.22 GB (-13.6%)** |
| Inference time | 23.2s | **22.1s** | **-1.1s (1.05x faster)** |
| Resolution | 768×1024 | 768×1024 | same |
| Quality | reference | identical | no loss |
| Optimisations | none | xformers + VAE tiling | — |

**Key finding:** Saved 2.22 GB of VRAM with zero quality loss and slightly
faster inference — purely by changing how attention is computed internally.

---

## What We Changed (and Why)

### Change 1: xformers Memory-Efficient Attention

**The problem with standard attention:**

In the UNet, every image patch "attends" to every other image patch.
This creates an n×n attention matrix where n = number of patches.

For our 768×1024 image at patch size 8:
- Latent size: 96 × 128 = 12,288 patches
- Attention matrix: 12,288 × 12,288 = 150 million values
- Memory per layer (float16): 150M × 2 bytes = **300 MB**
- The UNet has ~32 attention layers: 300 MB × 32 = **~9.6 GB just for attention**

This scales quadratically: double the resolution → 4× the attention memory.

**How xformers fixes it:**

xformers implements FlashAttention, which computes the same result but
never builds the full n×n matrix. Instead it processes the query, key,
and value matrices in blocks that fit in fast on-chip GPU cache (SRAM).

```
Standard attention:  O(n²) memory — must store entire n×n matrix in VRAM
xformers attention:  O(n)  memory — processes in blocks, nothing stored
```

The output is **mathematically identical** — this is not an approximation.
It is a more efficient algorithm for the exact same computation.

**The code change (2 lines):**

```python
# Step 0 baseline — nothing here, uses standard attention by default

# Step 1 addition:
pipe.enable_xformers_memory_efficient_attention()
pipe.unet_encoder.enable_xformers_memory_efficient_attention()
```

This iterates through every attention layer in both UNets and replaces
the attention processor with the xformers version. Nothing else changes.

**Why the saving is 2.22 GB and not the full ~9.6 GB:**

The 9.6 GB is a peak calculation assuming all layers are in memory at once.
PyTorch uses a memory pool and reuses allocations between layers, so the
actual peak savings are smaller. However, the saving grows significantly
at higher resolutions (which is why Step 2 becomes possible).

---

### Change 2: VAE Tiling

**The problem:**

The VAE decoder upsamples the 96×128 latent back to 768×1024.
During upsampling, intermediate feature maps must be held in memory.
At 768×1024, this costs ~1-2 GB at peak.

**How tiling fixes it:**

`pipe.vae.enable_tiling()` splits the latent into overlapping tiles,
decodes each tile separately, and stitches them back.

```
Without tiling:  full 96×128 latent → all intermediate maps → 768×1024 image
With tiling:     tile 1 → stitch | tile 2 → stitch | ... → 768×1024 image
```

Peak memory per tile is tiny. Tiles overlap to avoid visible seams.

**The code change (1 line):**

```python
pipe.vae.enable_tiling()
```

---

## Why This Matters for the PhD Research

The xformers result demonstrates a core principle of memory optimisation:

> **Algorithm choice can reduce memory complexity class, not just constant factors.**

Standard attention is O(n²). xformers is O(n). This is not a micro-optimisation
— it is a fundamental algorithmic improvement that scales dramatically with input size.

For the Edinburgh PhD on "automatically mapping models onto efficient spatial
systems," the research question this raises is:

*Can a compiler pass automatically identify attention operations in a model
graph and substitute them with memory-efficient implementations?*

This is exactly what a smart memory-aware compiler would do — profile the
model graph, identify quadratic memory operations, and replace them with
linear equivalents automatically.

---

## For the Website Backend

At 14.07 GB peak VRAM:
- The RTX 3090 (24 GB) has 9.9 GB free alongside this model
- A web server process typically uses ~0.5-1 GB
- Comfortable headroom for concurrent requests

---

## Output

- `results/tryon_output.png` — try-on result (visually identical to baseline)
- `results/human_input.png` — input person photo
- `results/garment_input.png` — input garment photo
- `results/mask_preview.png` — masked region
- `results/metrics.json` — full benchmark data

---

## What Is Next

Step 1 saved 2.22 GB through algorithmic improvement alone.
Step 2 applies **INT8 quantisation** — compressing the model weights
themselves from float16 (2 bytes) to int8 (1 byte), targeting a further
~6-7 GB reduction.

Combined: the same 768×1024 image could run in ~7-8 GB VRAM,
making it viable on any GPU with 8+ GB — the most common consumer tier.

---

*Previous: [Step 0 — Baseline](../step0_baseline/REPORT.md)*
*Next: [Step 2 — INT8 Quantisation](../step2_quantization/REPORT.md)*
