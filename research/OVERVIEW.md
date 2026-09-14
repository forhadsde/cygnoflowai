# Memory Optimisation for Generative AI — Research Overview

**Project:** IDM-VTON Virtual Try-On — Memory Optimisation Study  
**Hardware:** NVIDIA RTX 3090 (24 GB VRAM)  
**Researcher:** Md Forhadul Islam  
**Target:** PhD Application — University of Edinburgh  
**Position:** Memory Optimisation for Distributed ML Systems (Dr Jianyi Cheng)

---

## 1. What This Project Is About

This project investigates how to run large AI image generation models more
efficiently — using less GPU memory (VRAM) while maintaining or improving
output quality.

The model under study is **IDM-VTON**: a state-of-the-art virtual clothing
try-on system. Given a photo of a person and a photo of a garment, it
generates a realistic image of the person wearing that garment. It is built
on top of Stable Diffusion XL (SDXL), one of the most powerful open-source
image generation architectures.

The same problem — how do we fit increasingly large models into limited
memory — is the central challenge of modern ML systems research. This is
exactly the research area of the Edinburgh PhD position.

---

## 2. Why Memory Is the Bottleneck

Modern AI image models are enormous:

| Component | Size (float16) |
|---|---|
| Main UNet (denoiser) | ~6 GB |
| Garment UNet encoder | ~5 GB |
| VAE (image encoder/decoder) | ~0.8 GB |
| CLIP text encoders | ~0.5 GB |
| **Total** | **~13–14 GB** |

Most consumer GPUs have 8–12 GB of VRAM. Running all components
simultaneously on a single GPU requires either expensive hardware or
clever memory management strategies.

This project benchmarks and compares different memory strategies
systematically, producing measurable results at each step.

---

## 3. Research Question

> *Can we automatically apply memory optimisation techniques to large
> generative models to reduce VRAM usage while maintaining output quality
> — and can these techniques be composed to achieve multiplicative savings?*

This maps directly to the Edinburgh PhD project:
"develop techniques that automatically map emerging models onto
efficient spatial systems."

---

## 4. Experiment Roadmap

Each step is a controlled experiment. Only one variable changes at a time
so results are cleanly attributable.

```
Step 0 — Baseline (DONE)
         All models on GPU, no optimisation
         768×1024, 16.29 GB VRAM, 23.2s inference

Step 1 — xformers Memory-Efficient Attention
         Replace standard O(n²) attention with O(n) attention
         Same resolution. Expected: ~10–12 GB VRAM, similar speed
         Technique: xformers library + VAE tiling

Step 2 — INT8 Weight Quantisation
         Reduce weight precision from float16 → int8 (half the size)
         Same resolution. Expected: ~7–9 GB VRAM, potentially faster
         Technique: bitsandbytes dynamic quantisation

Step 3 — Combined (Production Version)
         xformers + INT8 together
         Expected: ~6–8 GB VRAM — fits on a budget 8 GB GPU
         This becomes the backend API for the website
```

---

## 5. Connection to PhD Research

The Edinburgh position focuses on:
- Memory optimisation for distributed ML systems
- Automatically mapping models onto efficient spatial systems
- CUDA-level systems programming

This project provides:
- Concrete benchmarking methodology for memory strategies
- Empirical data on composability of optimisation techniques
- Real-world motivation: a production AI service that must run on constrained hardware
- CUDA experience: all experiments use PyTorch CUDA with direct GPU memory profiling

The natural PhD extension of this work would be to automate the
selection and composition of these techniques — building a compiler
pass or runtime system that profiles any model and applies the optimal
memory strategy without manual intervention.

---

## 6. Folder Structure

```
research/
  OVERVIEW.md                    ← this file
  step0_baseline/
    script.py                    ← annotated baseline inference
    REPORT.md                    ← full explanation + results
    results/                     ← output images + metrics.json
  step1_xformers/
    script.py                    ← xformers + VAE tiling
    REPORT.md                    ← generated after running
    results/
  step2_quantization/
    script.py                    ← INT8 quantisation
    REPORT.md                    ← generated after running
    results/
  step3_combined/
    script.py                    ← all techniques together
    REPORT.md                    ← generated after running
    results/
  FINAL_COMPARISON.md            ← all results side by side
```

---

## 7. Product Goal

The final optimised version (Step 3) will serve as the inference backend
for a virtual try-on web service. Requirements:

- VRAM: must fit within 8–10 GB (shared with web server)
- Latency: under 30 seconds per request
- Quality: output must be visually equivalent to the baseline

These are real engineering constraints, not academic ones. Meeting them
with measured, documented experiments is the core of the project.

---

*All experiments run on Windows 11, RTX 3090 24 GB, PyTorch 2.1.0 + CUDA 12.1.*
