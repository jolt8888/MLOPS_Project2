# Quick Start Script - Example training runs

# 1. Basic training with default parameters
python train.py

# 2. Train with custom learning rate and checkpoint directory
python train.py --checkpoint_dir models --lr 1e-3

# 3. Train for more epochs with different batch size
python train.py --epochs 5 --train_batch_size 16 --eval_batch_size 16

# 4. Train on different GLUE task (SST-2 sentiment classification)
python train.py --task_name sst2 --lr 3e-5 --epochs 3

# 5. Full configuration example
python train.py `
    --model_name_or_path distilbert-base-uncased `
    --task_name mrpc `
    --lr 2e-5 `
    --epochs 3 `
    --train_batch_size 32 `
    --eval_batch_size 32 `
    --max_seq_length 128 `
    --warmup_steps 100 `
    --weight_decay 0.01 `
    --checkpoint_dir checkpoints `
    --wandb_project glue-experiments `
    --wandb_run_name mrpc-baseline `
    --seed 42
