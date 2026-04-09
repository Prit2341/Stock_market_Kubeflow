# create_secret.ps1 — store your Alpha Vantage API key as a k8s secret
# Usage: .\pipeline\create_secret.ps1 -ApiKey "YOUR_KEY_HERE"
param([Parameter(Mandatory)][string]$ApiKey)

kubectl create secret generic alpha-vantage `
  --from-literal=api-key=$ApiKey `
  -n kubeflow `
  --dry-run=client -o yaml | kubectl apply -f -

Write-Host "Secret 'alpha-vantage' created in kubeflow namespace." -ForegroundColor Green
