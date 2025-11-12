FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install CPU-only PyTorch first (much smaller than default with CUDA)
RUN pip install --no-cache-dir torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install other Python dependencies (excluding torch since we installed it above)
RUN pip install --no-cache-dir datasets>=2.14.0 evaluate>=0.4.0 lightning>=2.0.0 transformers>=4.30.0 wandb>=0.15.0 scikit-learn>=1.3.0 scipy>=1.11.0

# Copy project files
COPY model.py .
COPY train.py .

# Create checkpoint directory
RUN mkdir -p /app/checkpoints

# Set environment variables
ENV PYTHONUNBUFFERED=1
# W&B mode can be set at runtime: -e WANDB_MODE=online (default) or -e WANDB_MODE=offline

# Default command runs training with best hyperparameters from Project 1
# Learning rate ≈ 2.33e-5, warmup ≈ 0.0966 (convert to steps)
CMD ["python", "train.py", \
     "--lr", "2.33e-5", \
     "--warmup_steps", "100", \
     "--epochs", "3", \
     "--train_batch_size", "32", \
     "--eval_batch_size", "32", \
     "--checkpoint_dir", "/app/checkpoints", \
     "--task_name", "mrpc", \
     "--wandb_project", "glue-docker-training", \
     "--wandb_run_name", "best-hyperparams"]
