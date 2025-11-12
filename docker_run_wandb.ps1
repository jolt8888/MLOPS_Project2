# Docker Build and Run Script with W&B Support
# This script builds and runs a Docker container with W&B logging enabled

param(
    [string]$WandbApiKey = $env:WANDB_API_KEY,
    [switch]$SkipBuild
)

# Build the Docker image (unless skipped)
if (-not $SkipBuild) {
    Write-Host "Building Docker image..." -ForegroundColor Green
    docker build -t glue-trainer:latest .
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nDocker build failed!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`nDocker image ready!" -ForegroundColor Green
Write-Host "`nStarting training with best hyperparameters:" -ForegroundColor Cyan
Write-Host "  - Learning rate: 2.33e-5" -ForegroundColor Yellow
Write-Host "  - Warmup steps: 100" -ForegroundColor Yellow
Write-Host "  - Epochs: 3" -ForegroundColor Yellow
Write-Host "  - Batch size: 32" -ForegroundColor Yellow

# Check for W&B API key
if ($WandbApiKey) {
    Write-Host "`n✓ W&B logging ENABLED" -ForegroundColor Green
    Write-Host "  Project: glue-docker-training" -ForegroundColor Cyan
    Write-Host "  Run name: best-hyperparams-docker" -ForegroundColor Cyan
    Write-Host "  View at: https://wandb.ai/" -ForegroundColor Cyan
    Write-Host ""
    
    # Run with W&B enabled
    docker run --rm `
        --shm-size=2g `
        -v "${PWD}/checkpoints:/app/checkpoints" `
        -e WANDB_API_KEY=$WandbApiKey `
        -e WANDB_MODE=online `
        glue-trainer:latest `
        python train.py `
        --lr 2.33e-5 `
        --warmup_steps 100 `
        --epochs 3 `
        --train_batch_size 32 `
        --eval_batch_size 32 `
        --checkpoint_dir /app/checkpoints `
        --task_name mrpc `
        --wandb_project glue-docker-training `
        --wandb_run_name best-hyperparams-docker
} else {
    Write-Host "`n⚠ W&B logging DISABLED (no API key found)" -ForegroundColor Yellow
    Write-Host "`nTo enable W&B logging:" -ForegroundColor Yellow
    Write-Host '  1. Get your API key from: https://wandb.ai/settings' -ForegroundColor Cyan
    Write-Host '  2. Set environment variable: $env:WANDB_API_KEY = "your_key"' -ForegroundColor Cyan
    Write-Host '  3. Re-run this script: .\docker_run_wandb.ps1 -SkipBuild' -ForegroundColor Cyan
    Write-Host ""
    
    # Run in offline mode
    docker run --rm `
        --shm-size=2g `
        -v "${PWD}/checkpoints:/app/checkpoints" `
        -e WANDB_MODE=offline `
        glue-trainer:latest
}
