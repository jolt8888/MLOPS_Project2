FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY model.py .
COPY train.py .

# Create checkpoint directory
RUN mkdir -p /app/checkpoints

# Set environment variables
ENV PYTHONUNBUFFERED=1
# Disable W&B for faster training without logging overhead (override with -e WANDB_MODE=online)
ENV WANDB_MODE=disabled

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
