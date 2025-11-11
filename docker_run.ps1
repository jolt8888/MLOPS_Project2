# Docker Build and Run Script
# This script builds and runs a Docker container with the best hyperparameters from Project 1

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
    Write-Host ""
    
    # Run the container
    # Mount local checkpoints directory to save models
    # --shm-size increases shared memory (helps with multiprocessing)
    # --storage-opt size=10G increases container storage (requires specific storage drivers)
    docker run --rm `
        --shm-size=2g `
        -v "${PWD}/checkpoints:/app/checkpoints" `
        glue-trainer:latest
} else {
    Write-Host "`nDocker build failed!" -ForegroundColor Red
    exit 1
}
