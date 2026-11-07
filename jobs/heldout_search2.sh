#!/bin/bash
#SBATCH -J gpic-heldout-search
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:45:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Held-out loss of the rest of the Muon learning-rate search: the 6.8e-3 runs (d640 16704/33408, d1024 5024/10048;
# runs/gpic-speed/muon_lr_search/mlr6p8e-3) and the wide search on the final ladder code (d1280 2800/5600, d1792 1120/2240 at
# 3.4e-3, 6.8e-3, 1.36e-2; runs/gpic-speed/muon_lr_search2/mlr<tag>), EMA and raw, N 3072 + 3072, appended to
# tmp/heldout_loss/heldout_loss.jsonl. At these short runs the EMA (0.9996) lags the raw weights (d1024 at 10048: 4.5%), so the
# raw loss is the cleaner lr signal. train_flops in the output assumes batch 256 (GLOBAL_BATCH); these runs use 1024.
set -eo pipefail
WC=/scratch/gpfs/GROUP/USER
WT=$WC/project/gpic-wt/heldout_loss
source $WC/project/gpic/jobs/env.sh
export PYTHONPATH=$WT/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source $WT/jobs/heldout_loss_lib.sh
R=$WC/runs/gpic-speed/muon_lr_search R2=$WC/runs/gpic-speed/muon_lr_search2
CK=($R/mlr6p8e-3/exp_ladder_d640/ms-step=016704.ckpt $R/mlr6p8e-3/exp_ladder_d640/ms-step=033408.ckpt
    $R/mlr6p8e-3/exp_ladder_d1024/ms-step=005024.ckpt $R/mlr6p8e-3/exp_ladder_d1024/ms-step=010048.ckpt)
for t in mlr3p4e-3 mlr6p8e-3 mlr1p36e-2; do
  CK+=($R2/$t/exp_ladder_d1280/ms-step=002800.ckpt $R2/$t/exp_ladder_d1280/ms-step=005600.ckpt $R2/$t/exp_ladder_d1792/ms-step=001120.ckpt $R2/$t/exp_ladder_d1792/ms-step=002240.ckpt)
done
for c in "${CK[@]}"; do [ -f "$c" ] || { echo "missing $c"; exit 2; }; done
heldout_run $WC/tmp/heldout_loss/heldout_loss.jsonl 3072 3072 8 "${CK[@]}"
