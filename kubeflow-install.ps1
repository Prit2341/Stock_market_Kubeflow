# Kubeflow Full Reset & Install Script
Write-Host "=== Step 1: Tearing down existing Kubeflow ===" -ForegroundColor Cyan

kubectl delete namespace kubeflow --ignore-not-found=true
Start-Sleep -Seconds 10

# Delete all PVs related to kubeflow
kubectl get pv | Select-String "kubeflow" | ForEach-Object {
    $pvName = ($_ -split '\s+')[0]
    kubectl delete pv $pvName --force --grace-period=0
}

Write-Host "=== Step 2: Cleaning storage on master node ===" -ForegroundColor Cyan
docker exec k3d-stockcluster-server-0 sh -c "rm -rf /var/lib/rancher/k3s/storage/*"

Write-Host "=== Step 3: Installing cluster-scoped resources ===" -ForegroundColor Cyan
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.16.0"
Start-Sleep -Seconds 5

Write-Host "=== Step 4: Installing Kubeflow Pipelines ===" -ForegroundColor Cyan
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=2.16.0"

Write-Host "=== Step 5: Waiting for pods to start (3 min) ===" -ForegroundColor Cyan
Start-Sleep -Seconds 180

Write-Host "=== Step 6: Pod Status ===" -ForegroundColor Cyan
kubectl get pods -n kubeflow

Write-Host "=== Done! ===" -ForegroundColor Green
Write-Host "To access Kubeflow UI run:" -ForegroundColor Yellow
Write-Host "kubectl port-forward svc/ml-pipeline-ui 8080:80 -n kubeflow" -ForegroundColor Yellow
Write-Host "Then open: http://localhost:8080" -ForegroundColor Yellow
