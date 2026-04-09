# prepare_data.ps1
# Fetches stock data locally (where Yahoo Finance is accessible) and
# copies it into the k3d cluster PVC so the pipeline can use it.
# Run from repo root: .\pipeline\prepare_data.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent

# ── Step 1: Create PVC if it doesn't exist ───────────────────────────────────
Write-Host "=== Creating PVC 'stock-data' in kubeflow namespace ===" -ForegroundColor Cyan
kubectl apply -f "$PSScriptRoot\stock-data-pvc.yaml"

# ── Step 2: Fetch data locally ───────────────────────────────────────────────
Write-Host "`n=== Fetching stock data locally ===" -ForegroundColor Cyan
$localData = "$Root\pipeline\local_data"
New-Item -ItemType Directory -Force -Path "$localData\raw"     | Out-Null
New-Item -ItemType Directory -Force -Path "$localData\combined" | Out-Null

$env:DATA_DIR = $localData
python "$Root\components\1_fetch_data\src\fetch_stock_data.py"
if ($LASTEXITCODE -ne 0) { throw "Data fetch failed" }

# ── Step 3: Copy data into PVC via a temp pod ────────────────────────────────
Write-Host "`n=== Copying data into cluster PVC ===" -ForegroundColor Cyan

# Spin up a temp pod that mounts the PVC
$podYaml = @"
apiVersion: v1
kind: Pod
metadata:
  name: data-loader
  namespace: kubeflow
spec:
  restartPolicy: Never
  containers:
  - name: loader
    image: busybox
    command: ["sleep", "3600"]
    securityContext:
      runAsNonRoot: false
      runAsUser: 0
    volumeMounts:
    - name: data
      mountPath: /data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: stock-data
"@

$podYaml | kubectl apply -f -

Write-Host "Waiting for data-loader pod to be ready..."
kubectl wait pod/data-loader -n kubeflow --for=condition=Ready --timeout=60s

# Create directories in the pod first
kubectl exec -n kubeflow data-loader -- mkdir -p /data/raw /data/combined

# Copy each file individually
Write-Host "Copying raw CSVs..."
Get-ChildItem "$localData\raw" -Filter "*.csv" | ForEach-Object {
    kubectl cp $_.FullName "kubeflow/data-loader:/data/raw/$($_.Name)"
    Write-Host "  Copied $($_.Name)"
}

Write-Host "Copying combined CSV..."
Get-ChildItem "$localData\combined" -Filter "*.csv" | ForEach-Object {
    kubectl cp $_.FullName "kubeflow/data-loader:/data/combined/$($_.Name)"
    Write-Host "  Copied $($_.Name)"
}

# Clean up temp pod
kubectl delete pod data-loader -n kubeflow --ignore-not-found=true

Write-Host "`n=== Done! Data is loaded into the cluster PVC. ===" -ForegroundColor Green
Write-Host "Now run: cd pipeline && python submit_pipeline.py --run" -ForegroundColor Yellow
