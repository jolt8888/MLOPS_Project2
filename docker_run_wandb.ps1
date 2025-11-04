# Docker Build and Run Script with W&B Support
# This script builds and runs a Docker container with W&B logging enabled

param(
    [string]$WandbApiKey = $env:WANDB_API_KEY
)

# Build the Docker image
Write-Host "Building Docker image..." -ForegroundColor Green
docker build -t glue-trainer:latest .

# Check if build was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nDocker image built successfully!" -ForegroundColor Green
    Write-Host "`nStarting training with best hyperparameters:" -ForegroundColor Cyan
    Write-Host "  - Learning rate: 2.33e-5" -ForegroundColor Yellow
    Write-Host "  - Warmup ratio: 0.0966 (~100 steps)" -ForegroundColor Yellow
    Write-Host "  - Epochs: 3" -ForegroundColor Yellow
    Write-Host "  - Batch size: 32" -ForegroundColor Yellow
    Write-Host "`nNote: Training on CPU will take longer than GPU training on Colab" -ForegroundColor Magenta
    
    # Check for W&B API key
    if ($WandbApiKey) {
        Write-Host "`nW&B logging enabled" -ForegroundColor Green
        Write-Host ""
        
        # Run with W&B
        docker run --rm `
            -v "${PWD}/checkpoints:/app/checkpoints" `
            -e WANDB_API_KEY=$WandbApiKey `
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
            --wandb_run_name best-hyperparams
    } else {
        Write-Host "`nW&B logging disabled (no API key found)" -ForegroundColor Yellow
        Write-Host "To enable W&B, run: .\docker_run_wandb.ps1 -WandbApiKey YOUR_API_KEY" -ForegroundColor Yellow
        Write-Host ""
        
        # Run without W&B logging
        docker run --rm `
            -v "${PWD}/checkpoints:/app/checkpoints" `
            glue-trainer:latest
    }
} else {
    Write-Host "`nDocker build failed!" -ForegroundColor Red
    exit 1
}
