#!/bin/bash
#SBATCH -J gpic-fp8rep-eval
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:30:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# FD-DINOv2 and FID of a d16 FP8-check replicate at C0/4, C0/2 and C0 (genmodel's wsd_eval.sh steps, one 8-GPU evaluation after the other),
# then the held-out loss of all its ms checkpoints in the same job (gpic-28; the controller: no separate held-out job under 8 tasks).
# usage: fp8_rep_eval.sh <tag under runs/gpic-speed/muon_fp8recipe>
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
export NCCL_NVLS_ENABLE=0
source $WC/project/gpic/jobs/env.sh
TAG=${1:?tag}
for s in 3400 6800 13600; do RAW_SRC=muon_wsd/../muon_fp8recipe/$TAG python -I $WC/tmp/ladder_prof/make_raw_view.py d1536 $s; done
bash $WC/project/gpic-wt/eval_official_run2/jobs/evals_gmflow.sh 0 d1536 1536 16 4 $WC/runs/gpic-speed/raw_view/muon_fp8recipe/$TAG \
  --wt eval_official_run2 3400:euler:50 6800:euler:50 13600:euler:50
echo "FD rc=$?"
WT=$WC/project/gpic-wt/heldout_loss_run
export PYTHONPATH=$WT/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source $WT/jobs/heldout_loss_lib.sh
shopt -s nullglob
CK=($WC/runs/gpic-speed/muon_fp8recipe/$TAG/exp_ladder_d1536/ms-step=*.ckpt)
HELDOUT_TRAIN_SHARDS="00787" heldout_run $WC/tmp/heldout_loss/heldout_loss.jsonl 3072 3072 8 "${CK[@]}"
echo FP8_REP_EVAL_DONE
