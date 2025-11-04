# Docker Playground Quick Start Guide

## 1. Go to Docker Playground
Visit: https://labs.play-with-docker.com/
- Login with Docker Hub account
- Click "Start"
- Click "+ ADD NEW INSTANCE"

## 2. Copy All Files
You need to copy these files to the playground:
- Dockerfile
- model.py
- train.py
- requirements.txt

## 3. Create Files in Playground

### Method 1: Create each file manually

```bash
# Create project directory
mkdir glue-trainer && cd glue-trainer

# Create Dockerfile
cat > Dockerfile << 'EOF'
[Copy entire Dockerfile content here]
EOF

# Create requirements.txt
cat > requirements.txt << 'EOF'
datasets>=2.14.0
evaluate>=0.4.0
lightning>=2.0.0
torch>=2.0.0
transformers>=4.30.0
wandb>=0.15.0
EOF

# Create model.py (copy from your file)
cat > model.py << 'EOF'
[Copy entire model.py content here]
EOF

# Create train.py (copy from your file)
cat > train.py << 'EOF'
[Copy entire train.py content here]
EOF
```

### Method 2: Use GitHub (Recommended if you have a repo)

```bash
# Clone your repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

## 4. Build Docker Image

```bash
docker build -t glue-trainer:latest .
```

This will take several minutes as it downloads Python and all dependencies.

## 5. Run Training

### Option A: Quick test (no W&B)
```bash
docker run --rm glue-trainer:latest
```

### Option B: With W&B logging
```bash
docker run --rm \
  -e WANDB_API_KEY=your_wandb_api_key \
  -e WANDB_MODE=online \
  glue-trainer:latest
```

### Option C: Save checkpoints locally
```bash
mkdir -p checkpoints
docker run --rm -v $(pwd)/checkpoints:/app/checkpoints glue-trainer:latest
```

### Option D: Custom hyperparameters
```bash
docker run --rm glue-trainer:latest \
  python train.py --lr 1e-3 --epochs 2 --train_batch_size 16
```

## 6. Expected Output

You should see:
```
Loading mrpc dataset...
Initializing distilbert-base-uncased model...
Starting training for 3 epochs...
Epoch 0: 100%|██████████| 115/115 [XX:XX<00:00, X.XXit/s, loss=0.XXX]
Validation: 100%|██████████| 13/13 [XX:XX<00:00, X.XXit/s]
...
Training complete!
```

## 7. View Results in W&B

If you used W&B:
1. Go to https://wandb.ai/
2. Navigate to your project: "glue-docker-training"
3. View metrics, logs, and model artifacts

## Troubleshooting

### Out of memory
Reduce batch size:
```bash
docker run --rm glue-trainer:latest \
  python train.py --train_batch_size 8 --eval_batch_size 8
```

### Timeout in Playground (4 hour limit)
The playground sessions expire after 4 hours. For longer training:
- Use GitHub Codespaces instead
- Or run locally

### Slow download
The first time will be slow as it downloads:
- Python base image (~100MB)
- PyTorch (~800MB)
- Transformers models (~250MB)
- GLUE dataset (~10MB)

## GitHub Codespaces Alternative

1. Push your code to GitHub
2. Open repository on GitHub
3. Click "Code" → "Codespaces" → "Create codespace on main"
4. In the terminal:
   ```bash
   docker build -t glue-trainer:latest .
   docker run --rm glue-trainer:latest
   ```

Codespaces advantages:
- 60 hours free per month
- Better resources
- Persistent storage
- GPU available (paid tier)
