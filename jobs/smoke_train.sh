#!/bin/bash
#SBATCH -J gpic-smoke-train
#SBATCH --gres=gpu:8
#SBATCH --time=01:30:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#FETCHABLE
# Smoke: official GPIC PixelGen config on Lite shards 0-159, 1500 steps, val at 750/1500, ckpt at 1500.
set -eo pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
cd $PIXGEN
python main.py fit --config $REPO/baselines/PixelGen/configs_h100/lite_smoke.yaml --trainer.devices 8
