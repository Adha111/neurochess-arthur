param(
    [string]$Tag = "v3.0.1",
    [string]$Registry = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$LocalImage = "neuro-chess:$Tag"
$TargetImage = if ($Registry) { "$Registry/neuro-chess:$Tag" } else { $LocalImage }

Write-Host "[1/4] Building Docker image $LocalImage..." -ForegroundColor Cyan
docker build -t $LocalImage .

if ($Registry) {
    Write-Host "[2/4] Tagging and pushing $TargetImage..." -ForegroundColor Cyan
    docker tag $LocalImage $TargetImage
    docker push $TargetImage
} else {
    Write-Host "[2/4] Local image mode (no Registry specified)." -ForegroundColor Yellow
}

Write-Host "[3/4] Applying Kubernetes Kustomize manifests..." -ForegroundColor Cyan
kubectl apply -k k8s/

if ($Registry) {
    kubectl set image deployment/neuro-chess-deployment neuro-chess=$TargetImage -n neuro-chess
} else {
    kubectl rollout restart deployment/neuro-chess-deployment -n neuro-chess
}

Write-Host "[4/4] Waiting for Rolling Update..." -ForegroundColor Cyan
kubectl rollout status deployment/neuro-chess-deployment -n neuro-chess --timeout=180s
kubectl get pods,svc,hpa -n neuro-chess
