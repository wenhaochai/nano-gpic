#!/bin/bash
#SBATCH --job-name=gpic_full_jit_t2i_pixelgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=256G
#SBATCH --time=1-23:59:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --no-requeue

# =======================
# Environment
# =======================
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate pixgen

export PYTHONPATH=/path_to_your_code/baselines/PixelGen

# =======================
# Run
# =======================
cd path_to_your_code/baselines/PixelGen

mkdir -p logs

python main.py predict --config ./config_gpic_full_jit_sample.yaml --trainer.devices 8 --ckpt_path your_checkpoint_path.ckpt