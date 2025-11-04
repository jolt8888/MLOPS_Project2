# Quick GitHub Setup for Docker Playground
# This makes it much easier to run in Docker Playground or Codespaces

## Option 1: Push to GitHub (Recommended)

### 1. Create a new repository on GitHub
Go to https://github.com/new and create a new repository

### 2. Initialize Git in your project (if not already)
```powershell
git init
git add .
git commit -m "Add GLUE transformer training project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### 3. In Docker Playground or Codespaces
```bash
# Clone your repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Build and run
docker build -t glue-trainer:latest .
docker run --rm glue-trainer:latest
```

---

## Option 2: Manual File Creation in Docker Playground

If you don't want to use GitHub, copy each file manually:

### Step 1: Go to https://labs.play-with-docker.com/
- Login and start a new session
- Add a new instance

### Step 2: Create project directory
```bash
mkdir glue-trainer && cd glue-trainer
```

### Step 3: Create requirements.txt
```bash
cat > requirements.txt << 'EOF'
datasets>=2.14.0
evaluate>=0.4.0
lightning>=2.0.0
torch>=2.0.0
transformers>=4.30.0
wandb>=0.15.0
EOF
```

### Step 4: Create Dockerfile
```bash
cat > Dockerfile << 'EOF'
# Copy your Dockerfile content here
EOF
```

### Step 5: Create Python files
For `model.py` and `train.py`, you need to either:
- Use `vi` or `nano` to create files
- Or use `cat > filename << 'EOF'` and paste content

---

## Option 3: Use GitHub Codespaces (Better for this project)

GitHub Codespaces is better because:
- ✅ More resources (2 cores, 4GB RAM)
- ✅ 60 hours free per month
- ✅ Can install VS Code extensions
- ✅ Direct integration with your repo
- ✅ Better for longer training runs

### Steps:
1. Push your code to GitHub (see Option 1)
2. Go to your repository on GitHub
3. Click the green "Code" button
4. Select "Codespaces" tab
5. Click "Create codespace on main"
6. Wait for environment to load
7. In terminal:
   ```bash
   docker build -t glue-trainer:latest .
   docker run --rm glue-trainer:latest
   ```

---

## For W&B Logging in Cloud Environments

### Get your W&B API key:
1. Go to https://wandb.ai/settings
2. Copy your API key

### Use it in Docker:
```bash
docker run --rm \
  -e WANDB_API_KEY=your_key_here \
  -e WANDB_MODE=online \
  glue-trainer:latest
```

---

## Expected Differences Between Environments

### Local Machine vs Cloud:
1. **Speed**: Cloud (especially Playground) will be slower on CPU
2. **Resources**: Playground has limited RAM/CPU
3. **Performance**: Training metrics should be identical if:
   - Same random seed (✅ we use seed=42)
   - Same hyperparameters (✅ hardcoded in Dockerfile)
   - Same data (✅ downloaded from Hugging Face)

### Possible Adaptations Needed:
1. **Batch size**: May need to reduce if OOM errors
2. **Timeout**: Playground expires after 4 hours
3. **Network**: First run downloads ~1GB of data/models

---

## Quick Test Commands

### Test build:
```bash
docker build -t glue-trainer:latest .
```

### Test run (1 epoch only for quick test):
```bash
docker run --rm glue-trainer:latest python train.py --epochs 1 --train_batch_size 16
```

### Full training with W&B:
```bash
docker run --rm \
  -e WANDB_API_KEY=$YOUR_KEY \
  -e WANDB_MODE=online \
  glue-trainer:latest
```
