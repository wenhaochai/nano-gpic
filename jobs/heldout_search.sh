#!/bin/bash
#SBATCH -J gpic-heldout-search
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:45:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Held-out loss of the Muon learning-rate search (owner popup 2026-11-07 01:0x: FD and held-out loss): the ms checkpoints at steps/2
# and steps of the 6 runs (d640 16704/33408, d1024 5024/10048; runs/gpic-speed/muon_lr_search/mlr<tag>/exp_ladder_<size>), EMA and raw,
# N 3072 + 3072, appended to tmp/heldout_loss/heldout_loss.jsonl (the "ladder" field says lite for these roots; read run_root).
set -eo pipefail
WC=/scratch/gpfs/GROUP/USER
WT=$WC/project/gpic-wt/heldout_loss
source $WC/project/gpic/jobs/env.sh
export PYTHONPATH=$WT/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source $WT/jobs/heldout_loss_lib.sh
R=$WC/runs/gpic-speed/muon_lr_search
CK=()
for t in mlr8p5e-4 mlr1p7e-3 mlr3p4e-3; do
  CK+=($R/$t/exp_ladder_d640/ms-step=016704.ckpt $R/$t/exp_ladder_d640/ms-step=033408.ckpt $R/$t/exp_ladder_d1024/ms-step=005024.ckpt $R/$t/exp_ladder_d1024/ms-step=010048.ckpt)
done
for c in "${CK[@]}"; do [ -f "$c" ] || { echo "missing $c"; exit 2; }; done
heldout_run $WC/tmp/heldout_loss/heldout_loss.jsonl 3072 3072 8 "${CK[@]}"
