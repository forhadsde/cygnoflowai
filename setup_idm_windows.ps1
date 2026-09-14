# IDM-VTON Windows Setup for RTX 3090
# =====================================
# Installs Miniconda, creates the 'idm' environment, installs all dependencies,
# and downloads model checkpoints (DensePose, HumanParsing, OpenPose).
# The IDM-VTON diffusion model (~12 GB) downloads automatically on first inference.
#
# Run once as normal user (no admin required for user-local Miniconda):
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_idm_windows.ps1

$ErrorActionPreference = "Stop"

$ROOT      = Split-Path -Parent $MyInvocation.MyCommand.Path
$IDMVTON   = Join-Path $ROOT "IDM-VTON"
$CONDA_DIR = "$env:USERPROFILE\miniconda3"
$CONDA_EXE = "$CONDA_DIR\Scripts\conda.exe"
$ENV_NAME  = "idm"
$PY_EXE    = "$CONDA_DIR\envs\$ENV_NAME\python.exe"
$PIP_EXE   = "$CONDA_DIR\envs\$ENV_NAME\Scripts\pip.exe"

function Log($msg) { Write-Host "[SETUP] $msg" -ForegroundColor Cyan }
function OK($msg)  { Write-Host "[OK]    $msg" -ForegroundColor Green }
function Err($msg) { Write-Host "[ERR]   $msg" -ForegroundColor Red; exit 1 }

# ── Step 1: Install Miniconda ──────────────────────────────────────────────────
if (Test-Path $CONDA_EXE) {
    OK "Miniconda already installed at $CONDA_DIR"
} else {
    Log "Installing Miniconda3..."
    $installer = "$env:TEMP\Miniconda3.exe"
    if (-not (Test-Path $installer)) {
        Log "Downloading Miniconda3 installer..."
        Invoke-WebRequest `
            -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" `
            -OutFile $installer `
            -UseBasicParsing
    }
    Log "Running installer (silent)..."
    Start-Process -FilePath $installer `
        -ArgumentList "/InstallationType=JustMe", "/RegisterPython=0", "/S", "/D=$CONDA_DIR" `
        -Wait -NoNewWindow
    if (-not (Test-Path $CONDA_EXE)) { Err "Miniconda install failed" }
    OK "Miniconda installed"
}

# ── Step 2: Create conda environment ──────────────────────────────────────────
if (Test-Path "$CONDA_DIR\envs\$ENV_NAME") {
    OK "Conda env '$ENV_NAME' already exists"
} else {
    Log "Creating conda env '$ENV_NAME' with Python 3.10..."
    & $CONDA_EXE create -n $ENV_NAME python=3.10 -y
    if ($LASTEXITCODE -ne 0) { Err "conda create failed" }
    OK "Env created"
}

# ── Step 3: Install PyTorch 2.1 + CUDA 12.1 ───────────────────────────────────
$torchCheck = & $PY_EXE -c "import torch; print(torch.cuda.is_available())" 2>$null
if ($torchCheck -eq "True") {
    OK "PyTorch+CUDA already installed"
} else {
    Log "Installing PyTorch 2.1.0 with CUDA 12.1..."
    & $PIP_EXE install `
        "torch==2.1.0" "torchvision==0.16.0" "torchaudio==2.1.0" `
        --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) { Err "PyTorch install failed" }
    OK "PyTorch installed"
}

# ── Step 4: Install all Python dependencies ────────────────────────────────────
Log "Installing pip packages..."
& $PIP_EXE install --upgrade pip

$packages = @(
    "accelerate==0.25.0",
    "transformers==4.36.2",
    "diffusers==0.25.0",
    "einops==0.7.0",
    "scipy==1.11.1",
    "opencv-python",
    "tqdm==4.66.1",
    "gradio==4.24.0",
    "basicsr",
    "onnxruntime-gpu",
    "fvcore",
    "iopath",
    "cloudpickle",
    "omegaconf",
    "pycocotools",
    "portalocker",
    "av",
    "huggingface_hub"
)

foreach ($pkg in $packages) {
    Log "  Installing $pkg..."
    & $PIP_EXE install $pkg --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN]  $pkg failed, continuing..." -ForegroundColor Yellow
    }
}
OK "Pip packages installed"

# ── Step 5: Download model checkpoints ────────────────────────────────────────
Log "Downloading model checkpoints..."

# DensePose checkpoint (~253 MB)
$denseposeDir = Join-Path $IDMVTON "ckpt\densepose"
$denseposeFile = Join-Path $denseposeDir "model_final_162be9.pkl"
if (-not (Test-Path $denseposeFile)) {
    New-Item -ItemType Directory -Force -Path $denseposeDir | Out-Null
    Log "Downloading DensePose checkpoint (~253 MB)..."
    Invoke-WebRequest `
        -Uri "https://dl.fbaipublicfiles.com/detectron2/DensePose_COCO/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl" `
        -OutFile $denseposeFile `
        -UseBasicParsing
    OK "DensePose checkpoint downloaded"
} else {
    OK "DensePose checkpoint already present"
}

# Human parsing ONNX models (~100 MB each)
$parsingDir = Join-Path $IDMVTON "ckpt\humanparsing"
New-Item -ItemType Directory -Force -Path $parsingDir | Out-Null

$parsingFiles = @{
    "parsing_atr.onnx" = "https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/parsing_atr.onnx"
    "parsing_lip.onnx" = "https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/parsing_lip.onnx"
}

foreach ($entry in $parsingFiles.GetEnumerator()) {
    $dest = Join-Path $parsingDir $entry.Key
    if (-not (Test-Path $dest)) {
        Log "Downloading $($entry.Key)..."
        Invoke-WebRequest -Uri $entry.Value -OutFile $dest -UseBasicParsing
        OK "$($entry.Key) downloaded"
    } else {
        OK "$($entry.Key) already present"
    }
}

# OpenPose ckpt dir (body_pose_model.pth auto-downloads on first run via basicsr)
$openposeDir = Join-Path $IDMVTON "ckpt\openpose\ckpts"
New-Item -ItemType Directory -Force -Path $openposeDir | Out-Null
OK "OpenPose ckpt dir ready (body_pose_model.pth auto-downloads on first run)"

# ── Step 6: Verify GPU and imports ────────────────────────────────────────────
Log "Verifying installation..."

$verifyScript = @"
import sys, os
ROOT    = r'$ROOT'
IDMVTON = r'$IDMVTON'
DEMO    = os.path.join(IDMVTON, 'gradio_demo')
sys.path.insert(0, IDMVTON)
sys.path.insert(0, DEMO)

import torch
print(f'PyTorch  : {torch.__version__}')
print(f'CUDA ok  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU      : {torch.cuda.get_device_name(0)}')
    print(f'VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

import detectron2; print(f'detectron2: {detectron2.__version__}')
import diffusers;   print(f'diffusers : {diffusers.__version__}')
import transformers; print(f'transformers: {transformers.__version__}')
import onnxruntime; print(f'onnxruntime: {onnxruntime.__version__}')
print('All imports OK')
"@

$verifyScript | & $PY_EXE
if ($LASTEXITCODE -ne 0) { Err "Verification failed - check errors above" }

# ── Done ───────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Run inference with:" -ForegroundColor White
Write-Host "    $CONDA_DIR\envs\$ENV_NAME\python.exe tryon_3090.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "  NOTE: First run downloads the IDM-VTON diffusion model" -ForegroundColor White
Write-Host "  (~12 GB from HuggingFace) - this takes 10-30 minutes." -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Green
