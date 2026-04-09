# build_images.ps1
# Builds all pipeline Docker images and loads them into the k3d cluster.
# Run this from the repo root: .\pipeline\build_images.ps1

param(
    [string]$ClusterName = "stockcluster"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent

Write-Host "=== Building pipeline component images ===" -ForegroundColor Cyan

$components = @(
    @{ Tag = "stock-fetch-data:latest";                Context = "components/1_fetch_data" },
    @{ Tag = "stock-feature-engineering:latest";       Context = "components/2_feature_engineering" },
    @{ Tag = "stock-train-xgboost:latest";             Context = "components/3_train_xgboost" },
    @{ Tag = "stock-train-gradient-boosting:latest";   Context = "components/3_train_gradient_boosting" },
    @{ Tag = "stock-train-random-forest:latest";       Context = "components/3_train_random_forest" },
    @{ Tag = "stock-model-evaluation:latest";          Context = "components/4_model_evaluation" }
)

foreach ($c in $components) {
    Write-Host "`n>>> Building $($c.Tag)" -ForegroundColor Yellow
    docker build -t $c.Tag "$Root/$($c.Context)"
    if ($LASTEXITCODE -ne 0) { throw "Build failed for $($c.Tag)" }
}

Write-Host "`n=== Loading images into k3d cluster '$ClusterName' ===" -ForegroundColor Cyan

$tags = $components | ForEach-Object { $_.Tag }
k3d image import @tags -c $ClusterName

if ($LASTEXITCODE -ne 0) { throw "k3d image import failed" }

Write-Host "`n=== Done! All images loaded into k3d ===" -ForegroundColor Green
Write-Host "Next step: cd pipeline && python submit_pipeline.py" -ForegroundColor Yellow
