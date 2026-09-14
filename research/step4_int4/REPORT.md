# Step 4 -- INT4 Weight Quantisation (W4A16)

**Method:** Compress weights from int8 (1 byte) to int4 (0.5 bytes) via bit packing. Combined with xformers from Step 1.
**Resolution:** 768 x 1024
**Hardware:** NVIDIA RTX 3090 (24 GB VRAM)

---

## Results vs All Previous Steps

| Metric | Step 0 Baseline | Step 2 INT8 | Step 4 INT4 | Change vs Baseline |
|---|---|---|---|---|
| VRAM peak | 16.29 GB | 11.29 GB | **9.06 GB** | **-7.23 GB (-44.4%)** |
| Inference time | 23.9s | 24.7s | 27.5s | +3.6s (unpack overhead) |
| Resolution | 768x1024 | 768x1024 | 768x1024 | same |
| Quality | reference | near-identical | acceptable | slight softening |
| Weight memory | 10,592 MB | 5,941 MB | **3,612 MB** | **-6,980 MB (-65.9%)** |

**Key finding:** INT4 packing compresses 1,646 linear layers from 0.5 bytes/weight
down to 0.25 bytes/weight (two values packed per byte). Weight memory dropped from
10,592 MB (baseline fp16) to 3,612 MB -- a 65.9% reduction. Peak VRAM hit 9.06 GB,
saving 7.23 GB from baseline and a further 2.23 GB over INT8.

**Quality verdict: PASS for product try-on.** The garment shape, colour, pockets,
and buttons all transferred correctly. Fine texture detail is slightly softer than
INT8 but the output is commercially acceptable for an e-commerce try-on product.

---

## What We Changed and Why

### From INT8 to INT4

In Step 2 we stored each weight as one int8 byte (range -127 to 127, 255 levels).
INT4 stores each weight in 4 bits (range -7 to 7, 15 levels) -- exactly half the bits.

| Format | Bits | Bytes | Levels | Max rounding error |
|---|---|---|---|---|
| float16 | 16 | 2.0 | continuous | ~0 |
| int8 | 8 | 1.0 | 255 | scale / 254 |
| int4 | 4 | 0.5 | 15 | scale / 14 |

The rounding error per weight is ~18x larger in INT4 vs INT8 (1/14 vs 1/254 of the
scale). This is the fundamental trade-off.

---

## How Bit Packing Works

PyTorch has no native int4 dtype. We store two 4-bit values inside one int8 byte
using bitwise operations -- this is called nibble packing.

```python
# Quantise weight to [-7, 7]
w_q = (w / scale).round().clamp(-7, 7)

# Shift to unsigned [0, 14] (fits in 4 bits)
w_u = w_q + 7

# Pack: even columns -> low nibble, odd columns -> high nibble
low    = w_u[:, 0::2]           # values 0-14
high   = w_u[:, 1::2]           # values 0-14
packed = low | (high << 4)      # one byte holds two weights
```

At forward pass time:

```python
# Unpack
low  = packed & 0xF             # extract low nibble
high = (packed >> 4) & 0xF      # extract high nibble

# Interleave back into full matrix
w_u[:, 0::2] = low
w_u[:, 1::2] = high

# Unshift and dequantise
w_fp = (w_u - 7).to(fp16) * scale
output = input @ w_fp.T + bias
```

The unpack adds ~3s to inference time vs INT8 (interleave + two nibble extractions
per layer vs one multiply). This is the speed cost of INT4.

---

## Layers Quantised

| Model | Layers | Before | After | Saved |
|---|---|---|---|---|
| Main UNet | 905 | 5,705 MB | 1,909 MB | **3,796 MB** |
| Garment UNet encoder | 741 | 4,887 MB | 1,703 MB | **3,184 MB** |
| **Total** | **1,646** | **10,592 MB** | **3,612 MB** | **6,980 MB** |

For comparison, INT8 saved 4,652 MB from the same layers. INT4 saves 6,980 MB --
an additional 2,328 MB (2.27 GB) beyond INT8.

---

## Quality Analysis

INT4 uses only 15 discrete levels to represent each weight. Despite this aggressive
compression, the output quality was acceptable because:

1. **Per-channel scaling** adapts the scale to each output neuron independently,
   minimising the worst-case error for weights with diverse magnitudes.

2. **30-step denoising averages out errors** -- the diffusion process is inherently
   iterative. Small per-weight errors do not compound catastrophically because each
   denoising step re-anchors the latent via the noise schedule.

3. **Denim is a robust garment type** -- strong texture and high-contrast colour
   give the model strong conditioning signal that survives quantisation noise.

**Where INT4 would likely fail:**
- Garments with very subtle colour gradients (e.g. pastel ombre)
- Fine patterns (small text, tight plaid) that require high weight precision
- Higher guidance scales (> 4.0) which amplify quantisation artefacts

---

## Cumulative Optimisation Summary

| Step | Technique | VRAM peak | Saved from baseline | Mechanism |
|---|---|---|---|---|
| Step 0 | Baseline fp16 | 16.29 GB | -- | reference |
| Step 1 | xformers attention | 15.58 GB | 0.71 GB | O(n^2) -> O(n) attention |
| Step 2 | INT8 weights + xformers | 11.29 GB | 5.00 GB | 2 bytes -> 1 byte/weight |
| Step 4 | INT4 weights + xformers | **9.06 GB** | **7.23 GB** | 1 byte -> 0.5 bytes/weight |

---

## Trade-Off Summary

| Factor | INT8 (Step 2) | INT4 (Step 4) | Verdict |
|---|---|---|---|
| VRAM saving | 5.00 GB | 7.23 GB | INT4 wins |
| Quality | near-identical | acceptable | INT8 safer |
| Speed | 24.7s | 27.5s | INT8 faster |
| Implementation complexity | moderate | higher (bit packing) | INT8 simpler |
| Risk for complex garments | very low | moderate | INT8 safer |

**Recommendation:** Use INT8 (Step 2) for production. Use INT4 only when the
target GPU has less than 11 GB VRAM and quality trade-off is acceptable.

---

## What This Enables

At 9.06 GB peak VRAM, the model now runs on:
- RTX 3080 (10 GB) -- with ~1 GB headroom
- RTX 3070 Ti (8 GB) -- borderline, would need VAE offload for the final 1 GB
- RTX 4070 (12 GB) -- comfortable

---

## For the PhD Research

This experiment completes the precision trade-off curve:

```
Precision    : fp16   ->   int8   ->   int4
VRAM (GB)    : 16.29  ->  11.29   ->   9.06
Weight MB    : 10592  ->   5941   ->   3612
Quality      : ref    ->   ~same  ->   acceptable
```

The curve flattens -- going from fp16 to int8 cuts weight memory by 44%, but
going from int8 to int4 only cuts another 39% of the remaining weight (because
activations, KV cache, and other buffers are unchanged). This is the point of
diminishing returns that a memory-aware compiler would need to model.

The research question becomes:
*At what precision level does quality degrade unacceptably for a given garment
type, and can a compiler predict this threshold automatically from the model
weight distribution -- without running inference?*

---

## Output

- `results/tryon_output.png` -- try-on result (9.06 GB VRAM, 27.5s)
- `results/metrics.json` -- full benchmark data
- Compare with `../step2_quantization/results/tryon_output.png` for quality diff

---

*Previous: [Step 2 -- INT8 Quantisation](../step2_quantization/REPORT.md)*
