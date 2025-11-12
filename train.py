"""
Training script for GLUE benchmark fine-tuning with PyTorch Lightning and W&B logging.

Usage:
    python train.py --checkpoint_dir models --lr 1e-3 --epochs 3
"""
import argparse
import os
from datetime import datetime
from pathlib import Path

import lightning as L
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from model import GLUEDataModule, GLUETransformer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GLUE transformer model with PyTorch Lightning"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="distilbert-base-uncased",
        help="Pretrained model name or path (default: distilbert-base-uncased)",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="mrpc",
        choices=["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"],
        help="GLUE task name (default: mrpc)",
    )
    
    # Training arguments
    parser.add_argument(
        "--lr",
        "--learning_rate",
        type=float,
        default=2e-5,
        dest="learning_rate",
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Evaluation batch size (default: 32)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps (default: 0)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay (default: 0.0)",
    )
    
    # Checkpoint and logging arguments
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="glue-finetuning",
        help="W&B project name (default: glue-finetuning)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Accelerator type (default: auto)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices (default: 1)",
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed for reproducibility
    L.seed_everything(args.seed)
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B logger with API key if available
    import wandb
    wandb_api_key = os.getenv('WANDB_API_KEY')
    
    # Ensure W&B is logged in before creating logger
    if wandb_api_key and os.getenv('WANDB_MODE') != 'offline':
        # Write API key to config file for wandb to pick up
        wandb_dir = os.path.expanduser("~/.netrc")
        try:
            with open(wandb_dir, 'w') as f:
                f.write(f"machine api.wandb.ai\n")
                f.write(f"  login user\n")
                f.write(f"  password {wandb_api_key}\n")
            os.chmod(wandb_dir, 0o600)
            print("✓ W&B API key configured")
        except Exception as e:
            print(f"⚠ Failed to configure W&B: {e}")
    
    # Initialize W&B logger
    wandb_run_name = args.wandb_run_name or f"{args.task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=wandb_run_name,
        log_model=True,
    )
    
    # Log hyperparameters to W&B
    wandb_logger.experiment.config.update({
        "model_name": args.model_name_or_path,
        "task_name": args.task_name,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_seq_length": args.max_seq_length,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
    })
    
    # Initialize data module
    print(f"Loading {args.task_name} dataset...")
    dm = GLUEDataModule(
        model_name_or_path=args.model_name_or_path,
        task_name=args.task_name,
        max_seq_length=args.max_seq_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    dm.setup("fit")
    
    # Initialize model
    print(f"Initializing {args.model_name_or_path} model...")
    model = GLUETransformer(
        model_name_or_path=args.model_name_or_path,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    
    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}-{step}-{val_loss:.4f}",
        save_top_k=1,  # Only save the best checkpoint to save disk space
        monitor="val_loss",
        mode="min",
        save_last=False,  # Don't save last to save disk space
    )
    
    # Initialize trainer
    print(f"Starting training for {args.epochs} epochs...")
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )
    
    # Train the model
    trainer.fit(model, datamodule=dm)
    
    # Finish W&B run
    wandb.finish()
    
    print(f"\nTraining complete! Checkpoints saved to: {checkpoint_dir}")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"W&B run: {wandb_logger.experiment.url}")


if __name__ == "__main__":
    main()
