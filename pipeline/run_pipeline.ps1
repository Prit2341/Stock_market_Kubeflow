# run_pipeline.ps1
# Full end-to-end: build images → load into k3d → install KFP SDK → compile → upload → run
# Run from repo root: .\pipeline\run_pipeline.ps1

param(
    [switch]$SkipBuild,    # skip docker build + k3d import (images already loaded)
    [switch]$CompileOnly   # stop after compiling pipeline.yaml
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot   # run from pipeline/ dir

# ── Step 1: Build + load images ──────────────────────────────────────────────
if (-not $SkipBuild) {
    Write-Host "`n=== Step 1: Building and loading Docker images ===" -ForegroundColor Cyan
    .\build_images.ps1
} else {
    Write-Host "`n=== Step 1: Skipped (--SkipBuild) ===" -ForegroundColor DarkGray
}

# ── Step 2: Install KFP SDK ──────────────────────────────────────────────────
Write-Host "`n=== Step 2: Installing KFP SDK ===" -ForegroundColor Cyan
pip install -q -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# ── Step 3: Compile + upload + run ───────────────────────────────────────────
if ($CompileOnly) {
    Write-Host "`n=== Step 3: Compiling pipeline only ===" -ForegroundColor Cyan
    python submit_pipeline.py --compile-only
} else {
    Write-Host "`n=== Step 3: Compile + upload + run ===" -ForegroundColor Cyan
    # Check Kubeflow UI is reachable (works regardless of how port-forward was started)
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:8080" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
        Write-Host "[OK] Kubeflow UI reachable at http://localhost:8080" -ForegroundColor Green
    } catch {
        Write-Host "[WARN] Cannot reach http://localhost:8080. Is port-forward running?" -ForegroundColor Yellow
        Write-Host "       kubectl port-forward svc/ml-pipeline-ui 8080:80 -n kubeflow" -ForegroundColor Yellow
        exit 1
    }
    python submit_pipeline.py --run
}

Write-Host "`n=== Done! ===" -ForegroundColor Green
Write-Host "Kubeflow UI: http://localhost:8080" -ForegroundColor Yellow
