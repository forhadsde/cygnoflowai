"""
One-time conversion: unet diffusion_pytorch_model.bin → .safetensors (fp16)

Why: The .bin is stored in fp32 (~12 GB on disk).  Loading it into a Python
process creates a temporary fp32 copy in RAM before converting to fp16, causing
a ~18 GB peak that OOM-kills on a 15 GB WSL2 system.

With mmap=True (PyTorch 2.1+), the .bin file is memory-mapped — tensors are
backed by the file, not by RAM — so the fp32 data is never fully resident.
Converting each tensor to fp16 in-place uses only ~6 GB RAM.

The resulting .safetensors file is also memory-mapped at load time, so
from_pretrained(low_cpu_mem_usage=True) will never spike RAM again.

Run once from /home/rey/cygnoflow/:
    python3 convert_unet_to_safetensors.py
"""

import os, gc, json, shutil, time
import torch
from safetensors.torch import save_file

ROOT      = '/home/rey/cygnoflow'
UNET_DIR  = f'{ROOT}/IDM-VTON/ckpt/IDM-VTON/unet'
BIN_FILE  = f'{UNET_DIR}/diffusion_pytorch_model.bin'
ST_FILE   = f'{UNET_DIR}/diffusion_pytorch_model.safetensors'

def log(msg): print(f'[CONVERT] {msg}', flush=True)

if os.path.exists(ST_FILE):
    log(f'Already exists: {ST_FILE}')
    log('Delete it first if you want to re-convert.')
    exit(0)

if not os.path.exists(BIN_FILE):
    log(f'ERROR: {BIN_FILE} not found')
    exit(1)

log(f'Source : {BIN_FILE}  ({os.path.getsize(BIN_FILE)/1e9:.1f} GB)')
log('Loading with mmap=True — no full RAM copy of fp32 weights...')
t0 = time.time()

try:
    state_dict = torch.load(BIN_FILE, map_location='cpu', mmap=True, weights_only=True)
except TypeError:
    # weights_only not supported on very old torch; fall back
    log('weights_only=True failed, retrying without it...')
    state_dict = torch.load(BIN_FILE, map_location='cpu', mmap=True)

log(f'Loaded {len(state_dict)} tensors  ({time.time()-t0:.1f}s)')
log('Converting to fp16...')
t1 = time.time()

state_dict_fp16 = {k: v.to(torch.float16) for k, v in state_dict.items()}
del state_dict
gc.collect()
log(f'Conversion done  ({time.time()-t1:.1f}s)')

log(f'Saving safetensors → {ST_FILE}')
t2 = time.time()
save_file(state_dict_fp16, ST_FILE)
del state_dict_fp16
gc.collect()
log(f'Saved  ({time.time()-t2:.1f}s)')

# Update model_index.json (not required) and unet config if needed.
# No changes needed — diffusers auto-prefers .safetensors over .bin
# when both exist in the same directory.

elapsed = time.time() - t0
st_size = os.path.getsize(ST_FILE) / 1e9
log(f'Done in {elapsed:.1f}s  |  {st_size:.1f} GB on disk (fp16)')
log('You can now delete the .bin file to save 12 GB:')
log(f'  rm {BIN_FILE}')
