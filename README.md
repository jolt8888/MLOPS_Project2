# GLUE Transformer Training

Fine-tuning transformers on GLUE benchmark tasks with PyTorch Lightning and Weights & Biases logging.

## Installation

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

Make sure you're logged into Weights & Biases:

```bash
wandb login
```

Your training metrics will be automatically logged to W&B, including:
- Training loss
- Validation loss
- Task-specific metrics (accuracy, F1, etc.)
- Model checkpoints
- Hyperparameters

## Docker Usage

### Build and Run with Best Hyperparameters

Run training in Docker with the best hyperparameters from Project 1 (lr=2.33e-5, warmup=0.0966):

**Without W&B logging:**
```bash
.\docker_run.ps1
```

**With W&B logging:**
```bash
.\docker_run_wandb.ps1 -WandbApiKey YOUR_WANDB_API_KEY
```

Or set environment variable and run:
```bash
$env:WANDB_API_KEY = "your_api_key_here"
.\docker_run_wandb.ps1
```

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
├── model.py              # Model and data module definitions
├── train.py              # Training script with CLI
├── Dockerfile            # Docker image definition
├── .dockerignore         # Files to exclude from Docker build
├── docker_run.ps1        # Quick Docker build & run script
├── docker_run_wandb.ps1  # Docker run with W&B logging
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project configuration
├── example_runs.ps1      # Example training commands
├── README.md             # This file
└── checkpoints/          # Saved model checkpoints (created during training)
```
