#!/bin/bash
# All-in-one setup script for Docker Playground
# Copy and paste this entire script into Docker Playground terminal

echo "=== Creating project directory ==="
mkdir -p glue-trainer && cd glue-trainer

echo "=== Creating requirements.txt ==="
cat > requirements.txt << 'EOF'
datasets>=2.14.0
evaluate>=0.4.0
lightning>=2.0.0
torch>=2.0.0
transformers>=4.30.0
wandb>=0.15.0
EOF

echo "=== Creating Dockerfile ==="
cat > Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py .
COPY train.py .

RUN mkdir -p /app/checkpoints

ENV PYTHONUNBUFFERED=1
ENV WANDB_MODE=disabled

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
EOF

echo "=== Files created! ==="
echo "Next steps:"
echo "1. Create model.py and train.py files (copy from your local machine)"
echo "2. Run: docker build -t glue-trainer:latest ."
echo "3. Run: docker run --rm glue-trainer:latest"
