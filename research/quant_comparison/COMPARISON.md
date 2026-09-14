# Weight Quantisation Comparison — IDM-VTON Virtual Try-On

**Garment:** Blue denim shirt (company product)
**Model:** Same female model across all steps
**Resolution:** 768 x 1024
**Hardware:** NVIDIA RTX 3090 (24 GB VRAM)
**Note:** Steps 0–2 use no xformers — ensures colour accuracy. Steps 3–4 tested memory-saving combos; both show colour shift and are not recommended.

---

## Results at a Glance

| | Step 0 — Baseline | Step 1 — INT8 | Step 2 — INT4 | Step 3 — INT4 + Attn Slice | Step 4 — INT8 + xformers + upcast |
|---|---|---|---|---|---|
| **VRAM peak** | 16.28 GB | 11.70 GB | **9.30 GB** | 9.06 GB | 11.30 GB |
| **VRAM saved** | — | 4.58 GB | **6.98 GB** | 7.22 GB | 4.98 GB |
| **Inference time** | 22.9s | 25.4s | 28.5s | 42.0s | **23.6s** |
| **Weight memory** | 10,592 MB | 5,941 MB | **3,612 MB** | 3,612 MB | 5,941 MB |
| **Weight precision** | float16 (2 bytes) | int8 (1 byte) | int4 (0.5 bytes) | int4 (0.5 bytes) | int8 (1 byte) |
| **Colour accuracy** | Reference | Matches baseline | Matches baseline | Shifted (dark) | Shifted (dark) |
| **Recommended** | Reference | **Production** | 10 GB GPUs | Not recommended | Not recommended |
| **Garment quality** | Reference | Near-identical | Acceptable | Degraded | Near-identical |
| **Min GPU required** | 24 GB | 12 GB | 10 GB | 10 GB | 12 GB |

---

## Step 0 — Baseline (float16)

**VRAM: 16.28 GB | Time: 22.9s**

All model weights stored in float16 — 2 bytes per weight, full precision.
1,646 linear layers hold 10,592 MB of weight data.
This is the quality reference. Requires a 24 GB GPU (RTX 3090 / RTX 4090).

**Output:** `step0_baseline/results/tryon_output.png`

---

## Step 1 — INT8 Quantisation (W8A16)

**VRAM: 11.70 GB | Time: 25.4s | Saved: 4.58 GB (-28.1%)**

### What changed
Each weight compressed from float16 (2 bytes) to int8 (1 byte).
Activations remain float16 — only stored weights are compressed.

### How it works
```
scale[i] = max(|W[i,:]|) / 127.0        # one scale per output neuron
W_int8    = round(W / scale).clamp(-127, 127)  # 255 discrete levels

# At inference: dequantise just before matmul
W_fp16 = W_int8.float() * scale
output = input @ W_fp16.T + bias
```

### Result
- Weight memory: 10,592 MB → 5,941 MB (saved 4,651 MB)
- Colour: identical to baseline
- Quality: near-identical — 255 levels is sufficient for diffusion model weights
- Recommended for production use

**Output:** `step1_int8/results/tryon_output.png`

---

## Step 2 — INT4 Quantisation (W4A16)

**VRAM: 9.30 GB | Time: 28.5s | Saved: 6.98 GB (-42.9%)**

### What changed
Each weight compressed from float16 (2 bytes) to int4 (0.5 bytes) via bit packing.
Two 4-bit values are stored inside one byte using low/high nibbles.

### How it works
```
scale[i] = max(|W[i,:]|) / 7.0          # int4 range is -7 to +7 (15 levels)
W_int4    = round(W / scale).clamp(-7, 7)

# Pack two weights per byte
w_shifted = W_int4 + 7                  # [0, 14] fits in 4 bits
packed    = low_nibble | (high_nibble << 4)  # 0.5 bytes per weight

# At inference: unpack then dequantise
low   = packed & 0xF
high  = (packed >> 4) & 0xF
W_fp16 = (interleaved - 7).to(fp16) * scale
output = input @ W_fp16.T + bias
```

### Result
- Weight memory: 10,592 MB → 3,612 MB (saved 6,980 MB)
- Colour: identical to baseline
- Quality: acceptable — slight softening of fine texture, garment shape preserved
- Adds ~5.6s inference overhead vs INT8 (nibble unpacking per layer)
- Use when GPU has less than 12 GB VRAM

**Output:** `step2_int4/results/tryon_output.png`

---

## Trade-Off Analysis

### VRAM vs Quality

```
float16  ──────────────────────────  16.28 GB  ← full quality, needs 24 GB GPU
int8     ─────────────────           11.70 GB  ← near-identical, needs 12 GB GPU
int4     ──────────                   9.30 GB  ← acceptable, needs 10 GB GPU
```

### Precision levels per weight

| Format | Discrete levels | Max rounding error |
|---|---|---|
| float16 | ~65,000 | negligible |
| int8 | 255 | scale / 254 |
| int4 | 15 | scale / 14 |

INT4 rounding error is 18x larger than INT8 per weight. Across 30 denoising
steps this is visible as slight texture softening but not garment distortion.

### Speed

```
Baseline : 22.9s
INT8     : 25.4s  (+2.5s — dequantise fp16 per matmul)
INT4     : 28.5s  (+5.6s — unpack nibbles + dequantise per matmul)
```

### Recommendation by GPU

| GPU VRAM | Recommended config |
|---|---|
| 24 GB (RTX 3090 / 4090) | Baseline or INT8 |
| 12 GB (RTX 3080 Ti / 4070 Ti) | INT8 |
| 10 GB (RTX 3080 / 4070) | INT4 |
| 8 GB (RTX 3070 / 4060 Ti) | INT4 + VAE offload (future work) |

---

## Colour Consistency

All three outputs use **standard attention (no xformers)** which preserves
floating-point computation order across steps. The result is that all three
outputs show the same accurate denim blue — the garment colour from the
input photo is faithfully reproduced regardless of quantisation level.

This is the key design decision in this comparison: quantisation compresses
weight storage but does not change the attention computation path, so colour
drift does not occur.

---

## PhD Research Connection

This comparison demonstrates the **precision vs. resource trade-off curve**
for weight quantisation in diffusion-based generative models:

- Each step halves weight storage (2 bytes → 1 byte → 0.5 bytes)
- VRAM savings are significant but sub-linear (activations and KV cache unchanged)
- Quality degrades gracefully — each halving of precision is acceptable for
  product try-on use cases

The research question this raises:
> *Can a compiler automatically select the minimum precision for each layer
> that keeps output quality above an application-specific threshold?*

Some layers may tolerate int4 while others require int8. Per-layer mixed
precision (some layers int8, others int4) could achieve a better quality/VRAM
balance than uniform quantisation. This is the next natural experiment.

---

## Step 3 — INT4 + Attention Slicing (Not Recommended)

**VRAM: 9.06 GB | Time: 42.0s | Colour: shifted**

Attention slicing computes one attention head at a time instead of all heads
simultaneously. In theory this is mathematically identical to standard attention.
In practice, float16 arithmetic is not associative — computing in a different
order produces slightly different values. These differences compound over 30
denoising steps and produce a visible colour shift (same root cause as xformers).

Additional findings:
- VRAM saving over INT4 alone: only 0.24 GB (minimal — weight memory dominates)
- Speed penalty: 42.0s vs 28.5s (+48% slower)
- Colour: shifts to dark navy (not acceptable for product try-on)

**Conclusion: do not stack attention slicing on top of INT4.**
The trade-off is negative on all three axes: colour, speed, and VRAM.

---

## Step 4 — INT8 + xformers + upcast_attention (Not Recommended)

**VRAM: 11.30 GB | Time: 23.6s | Colour: shifted (dark)**

### Hypothesis
xformers causes colour drift because FlashAttention reorders float16 operations.
`upcast_attention=True` computes the attention score matrix in float32 before softmax,
which should eliminate the ordering sensitivity and allow xformers to be used
without colour drift.

### Why it failed
When xformers is enabled, it **replaces the entire attention kernel** with its own
CUDA implementation (FlashAttention). This completely bypasses the PyTorch-level
`upcast_attention` code path. The `upcast_attention=True` flag is simply never reached
during forward pass — xformers handles attention start-to-finish in its own kernel.

```
Standard attention path (upcast_attention works here):
  Q, K, V -> scores = Q @ K.T  [compute in float32] -> softmax -> scores @ V

xformers path (upcast_attention has no effect):
  Q, K, V -> [entire FlashAttention CUDA kernel, own precision handling] -> output
```

### Result
- Colour: shifted dark (same as xformers-only, hypothesis disproven)
- VRAM: 11.30 GB (slightly better than INT8 alone at 11.70 GB — xformers saves ~0.4 GB activation memory)
- Speed: 23.6s (faster than INT8 alone at 25.4s — xformers attention kernel is quicker)
- **Conclusion: upcast_attention cannot fix xformers colour drift. xformers is fundamentally
  incompatible with colour-accurate try-on inference.**

**Output:** `step4_upcast/results/tryon_output.png`

---

## PhD Research Connection (Extended)

Steps 3 and 4 demonstrate two failed optimisation paths, each for a different reason:

| Technique | Why it fails |
|---|---|
| Attention slicing | Changes float16 operation ORDER within PyTorch — errors compound |
| xformers + upcast_attention | xformers bypasses the Python attention path entirely — upcast_attention is a no-op |

This isolates the root cause precisely: **the colour drift is a CUDA kernel boundary issue,
not a precision issue**. The fix requires either (a) not using xformers, or (b) a custom
CUDA kernel that both uses FlashAttention memory layout AND accumulates in float32.
This is an open problem in efficient diffusion model deployment.

---

## Final Recommendation

For production use on a GPU server (cloud or own hardware):

| GPU VRAM | Use this config | Script |
|---|---|---|
| 24 GB | INT8 — near-identical quality | `step1_int8/script.py` |
| 12 GB | INT8 — comfortable headroom | `step1_int8/script.py` |
| 10 GB | INT4 — acceptable quality | `step2_int4/script.py` |
| < 10 GB | INT4 + model CPU offload (future work) | — |

---

## Files

```
quant_comparison/
  COMPARISON.md                    <- this document
  step0_baseline/
    script.py                      <- baseline inference script
    results/
      tryon_output.png             <- 16.28 GB VRAM, colour reference
      metrics.json
  step1_int8/
    script.py                      <- INT8 quantisation (RECOMMENDED)
    results/
      tryon_output.png             <- 11.70 GB VRAM, colour correct
      metrics.json
  step2_int4/
    script.py                      <- INT4 quantisation (10 GB GPUs)
    results/
      tryon_output.png             <- 9.30 GB VRAM, colour correct
      metrics.json
  step3_int4_attnslice/
    script.py                      <- INT4 + attention slicing (NOT recommended)
    results/
      tryon_output.png             <- 9.06 GB VRAM, colour shifted, 42s
      metrics.json
  step4_upcast/
    script.py                      <- INT8 + xformers + upcast_attention (NOT recommended)
    results/
      tryon_output.png             <- 11.30 GB VRAM, colour shifted, 23.6s
      metrics.json                 <- colour_accurate: false, explains why upcast fails with xformers
```
