import torch
import time

def log(msg):
    print(f"[LOG] {msg}")

def get_vram():
    return torch.cuda.max_memory_allocated() / 1024**3

def reset_vram():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

# ── Check GPU ──────────────────────────────────────────
log(f"GPU: {torch.cuda.get_device_name(0)}")
log(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

results = {}

# ── TEST 1: Baseline (no optimisation) ────────────────
log("\n=== TEST 1: Baseline ===")
reset_vram()
try:
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    )
    pipe = pipe.to("cuda")
    start = time.time()
    image = pipe(
        "a photo of a woman wearing a red dress",
        height=1024,
        width=1024,
        num_inference_steps=20
    ).images[0]
    elapsed = time.time() - start
    vram = get_vram()
    results["baseline"] = vram
    log(f"Baseline VRAM: {vram:.2f} GB")
    log(f"Baseline time: {elapsed:.1f}s")
    image.save("output_baseline.png")
except torch.cuda.OutOfMemoryError:
    vram = get_vram()
    results["baseline"] = "OOM"
    log(f"Baseline CRASHED with OOM at {vram:.2f} GB")
except Exception as e:
    results["baseline"] = "OOM"
    log(f"Baseline failed: {e}")

del pipe
torch.cuda.empty_cache()

# ── TEST 2: FP16 only ──────────────────────────────────
log("\n=== TEST 2: FP16 Mixed Precision ===")
reset_vram()
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    start = time.time()
    image = pipe(
        "a photo of a woman wearing a red dress",
        height=1024,
        width=1024,
        num_inference_steps=20
    ).images[0]
    elapsed = time.time() - start
    vram = get_vram()
    results["fp16"] = vram
    log(f"FP16 VRAM: {vram:.2f} GB")
    log(f"FP16 time: {elapsed:.1f}s")
    image.save("output_fp16.png")
except torch.cuda.OutOfMemoryError:
    results["fp16"] = "OOM"
    log(f"FP16 CRASHED with OOM")
except Exception as e:
    results["fp16"] = f"Error: {e}"
    log(f"FP16 failed: {e}")

del pipe
torch.cuda.empty_cache()

# ── TEST 3: FP16 + Attention Slicing ──────────────────
log("\n=== TEST 3: FP16 + Attention Slicing ===")
reset_vram()
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    start = time.time()
    image = pipe(
        "a photo of a woman wearing a red dress",
        height=1024,
        width=1024,
        num_inference_steps=20
    ).images[0]
    elapsed = time.time() - start
    vram = get_vram()
    results["fp16_attn"] = vram
    log(f"FP16+Attention VRAM: {vram:.2f} GB")
    log(f"FP16+Attention time: {elapsed:.1f}s")
    image.save("output_fp16_attn.png")
except torch.cuda.OutOfMemoryError:
    results["fp16_attn"] = "OOM"
    log(f"FP16+Attention CRASHED with OOM")
except Exception as e:
    results["fp16_attn"] = f"Error: {e}"
    log(f"FP16+Attention failed: {e}")

del pipe
torch.cuda.empty_cache()

# ── TEST 4: Full Optimisation ──────────────────────────
log("\n=== TEST 4: Full Optimisation (FP16 + Attention + VAE Tiling) ===")
reset_vram()
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()
    start = time.time()
    image = pipe(
        "a photo of a woman wearing a red dress",
        height=1024,
        width=1024,
        num_inference_steps=20
    ).images[0]
    elapsed = time.time() - start
    vram = get_vram()
    results["full_optimised"] = vram
    log(f"Full Optimised VRAM: {vram:.2f} GB")
    log(f"Full Optimised time: {elapsed:.1f}s")
    image.save("output_optimised.png")
except torch.cuda.OutOfMemoryError:
    results["full_optimised"] = "OOM"
    log(f"Full Optimised CRASHED with OOM")
except Exception as e:
    results["full_optimised"] = f"Error: {e}"
    log(f"Full Optimised failed: {e}")

# ── RESULTS TABLE ──────────────────────────────────────
log("\n========================================")
log("BENCHMARK RESULTS SUMMARY")
log("========================================")
log(f"{'Test':<30} {'Peak VRAM':>12}")
log("-" * 44)
for test, vram in results.items():
    if isinstance(vram, float):
        log(f"{test:<30} {vram:>10.2f} GB")
    else:
        log(f"{test:<30} {vram:>12}")

# Save to CSV
import csv
with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Test", "Peak VRAM (GB)"])
    for test, vram in results.items():
        writer.writerow([test, vram])

log("\nResults saved to results.csv")
log("Images saved as output_*.png")
