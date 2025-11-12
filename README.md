# GLUE Transformer Training - MLOPS Project 2

Fine-tuning DistilBERT on GLUE benchmark tasks (MRPC) using PyTorch Lightning with experiment tracking via Weights & Biases, containerized with Docker for reproducible training across environments.

## Project Overview

This project demonstrates MLOps best practices by converting a Jupyter notebook-based training workflow into a production-ready, containerized application with:
- **Modular Python scripts** with CLI argument parsing
- **Docker containerization** for reproducibility
- **Experiment tracking** with Weights & Biases
- **Best hyperparameters** from Project 1 (lr=2.33e-5, warmup=0.0966)
- **Cloud deployment** support (GitHub Codespaces, Docker Playground)

## Tasks Completed

### Task 2: Adapting Training Notebook to Python Scripts

The training code was restructured from Jupyter notebook format into modular Python scripts:

**`model.py`**: Contains model and data handling
- `GLUEDataModule`: Lightning DataModule for GLUE dataset loading and preprocessing
- `GLUETransformer`: Lightning Module with DistilBERT for sequence classification

**`train.py`**: Main training script with CLI
- Comprehensive argument parser for all hyperparameters
- W&B logger integration
- Model checkpointing (saves top 3 models)
- Single command training: `python train.py --checkpoint_dir models --lr 1e-3`

**Key improvements over notebook**:
- Reproducible with command-line arguments
- No manual cell execution required
- Proper logging and checkpointing
- Modular and maintainable code structure

### Task 3: Docker Containerization and Cloud Deployment

**Dockerfile optimizations**:
- Uses `python:3.10-slim` base image (smaller footprint)
- **CPU-only PyTorch** installation (reduces image size significantly for Docker Playground)
- Hardcoded best hyperparameters from Project 1
- W&B logging disabled by default for faster CPU training

**Testing environments**:
- Local Windows machine
- GitHub Codespaces (recommended)
- Docker Playground (with optimizations)

**Deployment process**:
1. Build: `docker build -t glue-trainer:latest .`
2. Run: `docker run --rm glue-trainer:latest`

See [CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md) for detailed instructions.

---

## Quick Start

### Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Or using pyproject.toml:

```bash
pip install -e .
```

## Usage

### Basic Training

Train with default parameters:

```bash
python train.py
```

### Custom Hyperparameters

Train with custom learning rate and checkpoint directory:

```bash
python train.py --checkpoint_dir models --lr 1e-3 --epochs 5
```

### Full Example

```bash
python train.py \
    --model_name_or_path distilbert-base-uncased \
    --task_name mrpc \
    --lr 2e-5 \
    --epochs 3 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --checkpoint_dir checkpoints \
    --wandb_project glue-finetuning \
    --wandb_run_name my-experiment \
    --seed 42
```

## Available Arguments

### Model Arguments
- `--model_name_or_path`: Pretrained model name (default: `distilbert-base-uncased`)
- `--task_name`: GLUE task (choices: `cola`, `sst2`, `mrpc`, `qqp`, `stsb`, `mnli`, `qnli`, `rte`, `wnli`)

### Training Arguments
- `--lr`, `--learning_rate`: Learning rate (default: `2e-5`)
- `--epochs`: Number of epochs (default: `3`)
- `--train_batch_size`: Training batch size (default: `32`)
- `--eval_batch_size`: Evaluation batch size (default: `32`)
- `--max_seq_length`: Maximum sequence length (default: `128`)
- `--warmup_steps`: Warmup steps (default: `0`)
- `--weight_decay`: Weight decay (default: `0.0`)

### Logging & Checkpointing
- `--checkpoint_dir`: Checkpoint directory (default: `checkpoints`)
- `--wandb_project`: W&B project name (default: `glue-finetuning`)
- `--wandb_run_name`: W&B run name (default: auto-generated)

### Other Arguments
- `--seed`: Random seed (default: `42`)
- `--accelerator`: Accelerator type (default: `auto`)
- `--devices`: Number of devices (default: `1`)

## W&B Setup

### Option 1: Set API Key as Environment Variable (Recommended for Docker)

```powershell
# Get your API key from https://wandb.ai/settings
$env:WANDB_API_KEY = "your_api_key_here"
```

### Option 2: Login via CLI

```bash
wandb login
```

Your training metrics will be automatically logged to W&B, including:
- **Training loss** - Loss during training
- **Validation loss** - Loss on validation set
- **Task-specific metrics** - Accuracy, F1, Precision, Recall
- **Model checkpoints** - Best model artifacts
- **Hyperparameters** - All training configuration
- **System metrics** - CPU/GPU usage, memory

View your experiments at: https://wandb.ai/

## Docker Usage

### Build and Run with Best Hyperparameters

Run training in Docker with the best hyperparameters from Project 1 (lr=2.33e-5, warmup=0.0966):

**Without W&B logging (offline mode):**
```powershell
.\docker_run.ps1
```

**With W&B logging (recommended):**
```powershell
# First, set your W&B API key (get it from https://wandb.ai/settings)
$env:WANDB_API_KEY = "your_api_key_here"

# Run training with W&B logging
.\docker_run_wandb.ps1

# Or skip rebuild if you've already built the image
.\docker_run_wandb.ps1 -SkipBuild
```

See [WANDB_SETUP.md](WANDB_SETUP.md) for detailed W&B setup instructions and performance verification guide.

### Manual Docker Commands

Build the image:
```bash
docker build -t glue-trainer:latest .
```

Run the container:
```bash
docker run --rm -v ${PWD}/checkpoints:/app/checkpoints glue-trainer:latest
```

Run with custom arguments:
```bash
docker run --rm -v ${PWD}/checkpoints:/app/checkpoints glue-trainer:latest `
    python train.py --lr 1e-3 --epochs 5
```

**Note:** Docker training will run on CPU by default, which is slower than GPU training on Colab.

## Project Structure

```
Project 2/
├── model.py                  # Model and data module definitions
├── train.py                  # Main training script with CLI
├── Dockerfile                # Optimized Docker image (CPU-only PyTorch)
├── .dockerignore             # Docker build optimization
├── docker_run.ps1            # Quick Docker build & run (Windows)
├── docker_run_wandb.ps1      # Docker run with W&B logging
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project configuration
├── example_runs.ps1          # Example training commands
├── README.md                 # This file
├── CLOUD_DEPLOYMENT.md       # Detailed cloud deployment guide
├── DOCKER_PLAYGROUND_GUIDE.md# Docker Playground specific instructions
└── checkpoints/              # Saved model checkpoints (created during training)
```

---

## Reflections

### What Went Well
- **Smooth notebook-to-script conversion**: Separation of model logic and training orchestration made code more maintainable
- **W&B integration**: Seamless experiment tracking with minimal code changes
- **Reproducibility**: Docker ensures consistent environment across machines
- **CLI design**: Comprehensive argparse setup makes hyperparameter tuning easy

### Challenges and Solutions

#### Challenge 1: Missing Dependencies
**Problem**: `evaluate` library failed with missing `scikit-learn` and `scipy` dependencies when running in Docker.

**Solution**: Added explicit dependencies to `requirements.txt`. This wasn't caught locally because these were already installed from previous projects.

**Lesson**: Always test in clean Docker environments to catch missing dependencies.

#### Challenge 2: Docker Playground Memory Constraints
**Problem**: Default PyTorch installation (~800MB with CUDA) is too large for Docker Playground's limited resources.

**Solution**: 
- Switched to CPU-only PyTorch (`torch==2.0.1+cpu`) which is ~200MB smaller
- Used `python:3.10-slim` instead of full Python image
- Added `.dockerignore` to exclude unnecessary files from build context

**Lesson**: Cloud environments have constraints; optimize for the deployment target.

#### Challenge 3: W&B Logging in Containers
**Problem**: W&B requires authentication which isn't available in containers by default.

**Solution**: 
- Set `WANDB_MODE=disabled` by default for faster CPU training
- Provide option to pass API key via environment variable: `-e WANDB_API_KEY=xxx`
- Created two run scripts: one without W&B, one with W&B support

**Lesson**: Make logging optional for testing, required for production.

### What I Would Improve Next Time

1. **Add unit tests**: Test data loading, model initialization, and training step functions
2. **CI/CD pipeline**: Automate Docker builds and tests with GitHub Actions
3. **Multi-GPU support**: Add distributed training configuration for cloud GPU instances
4. **Model registry**: Integrate W&B artifacts or MLflow for model versioning
5. **Configuration files**: Use YAML/JSON configs instead of CLI args for complex experiments
6. **Health checks**: Add endpoints to monitor training progress in containers
7. **Smaller test dataset**: Add option to train on subset for rapid testing

---

## Performance Verification

### Expected Results
Training on MRPC with best hyperparameters (lr=2.33e-5, warmup=0.0966, 3 epochs) should achieve:
- **Validation Accuracy**: ~85-88%
- **F1 Score**: ~88-90%

### Environment Comparison
| Environment | Performance | Speed | Notes |
|------------|-------------|-------|-------|
| Local (CPU) | Identical metrics | ~20 min/epoch | Baseline |
| Codespaces | Identical metrics | ~25 min/epoch | Slightly slower CPU |
| Docker Playground | Identical metrics | ~30+ min/epoch | Limited resources |

**Key finding**: Same seed (42) and hyperparameters produce identical results across all environments, confirming reproducibility.

---

## Docker Image Optimization

### Size Comparison
- **Before optimization**: ~2.5GB (PyTorch with CUDA support)
- **After optimization**: ~1.8GB (CPU-only PyTorch)
- **Improvement**: ~28% reduction

### Optimization Techniques Used
1. CPU-only PyTorch from wheel index
2. `--no-cache-dir` flag for pip
3. Slim Python base image
4. Single RUN command to reduce layers
5. `.dockerignore` to exclude unnecessary files

---

## Contributing

To add new features or improvements:
1. Fork the repository
2. Create a feature branch
3. Make changes and test with Docker
4. Submit a pull request

---

## License

This project is part of MLOPS coursework.

---

## Acknowledgments

- **Hugging Face** for transformers and datasets libraries
- **Lightning AI** for PyTorch Lightning framework
- **Weights & Biases** for experiment tracking
- **GLUE Benchmark** for evaluation tasks
