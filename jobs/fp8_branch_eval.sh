#!/bin/bash
#SBATCH -J gpic-fp8branch-eval
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:30:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The FP8 branch runs' last checkpoints (step 3400 = the d16 check's 13600; gpic-28, owner 2026-11-08 「可以」): FD-DINOv2 and FID of each
# (genmodel's wsd_eval.sh steps: a raw view, then the official evaluation in gpic-wt/eval_official_run2: Euler 50, official noise, CFG 1,
# 50K), one 8-GPU evaluation after the other, then the held-out loss of all of them in one 8-GPU pass (the controller: a held-out list under
# 8 tasks goes into the eval job). usage: fp8_branch_eval.sh <tag>...
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
export NCCL_NVLS_ENABLE=0
source $WC/project/gpic/jobs/env.sh
[ $# -ge 1 ] || { echo "usage: fp8_branch_eval.sh <tag>..."; exit 2; }
for TAG in "$@"; do
  RAW_SRC=muon_wsd/../muon_fp8branch/$TAG python -I $WC/tmp/ladder_prof/make_raw_view.py d1536 3400
  bash $WC/project/gpic-wt/eval_official_run2/jobs/evals_gmflow.sh 0 d1536 1536 16 4 $WC/runs/gpic-speed/raw_view/muon_fp8branch/$TAG \
    --wt eval_official_run2 3400:euler:50
  echo "$TAG FD rc=$?"
done
HW=$WC/project/gpic-wt/heldout_loss_run
export PYTHONPATH=$HW/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source $HW/jobs/heldout_loss_lib.sh
CK=(); for TAG in "$@"; do CK+=($WC/runs/gpic-speed/muon_fp8branch/$TAG/exp_ladder_d1536/ms-step=003400.ckpt); done
HELDOUT_TRAIN_SHARDS="00787" heldout_run $WC/tmp/heldout_loss/heldout_loss.jsonl 3072 3072 8 "${CK[@]}"
echo FP8_BRANCH_EVAL_DONE
