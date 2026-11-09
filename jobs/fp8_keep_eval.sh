#!/bin/bash
#SBATCH -J gpic-fp8keep-eval
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The FP8 resume test's two arms at C0 (step 13,600): FD-DINOv2 and FID of each (genmodel's wsd_eval.sh steps on eval_official_run3, the
# same results as run2 for the earlier d16 numbers), then the held-out loss of both in the same job.
# usage: fp8_keep_eval.sh <tag>...
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
export NCCL_NVLS_ENABLE=0
source $WC/project/gpic/jobs/env.sh
CK=()
for TAG in "$@"; do
  RAW_SRC=muon_wsd/../muon_fp8recipe/$TAG python -I $WC/tmp/ladder_prof/make_raw_view.py d1536 13600
  bash $WC/project/gpic-wt/eval_official_run3/jobs/evals_gmflow.sh 0 d1536 1536 16 4 $WC/runs/gpic-speed/raw_view/muon_fp8recipe/$TAG \
    --wt eval_official_run3 13600:euler:50
  echo "$TAG FD rc=$?"
  CK+=($WC/runs/gpic-speed/muon_fp8recipe/$TAG/exp_ladder_d1536/ms-step=013600.ckpt)
done
WT=$WC/project/gpic-wt/heldout_loss_run
export PYTHONPATH=$WT/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source $WT/jobs/heldout_loss_lib.sh
HELDOUT_TRAIN_SHARDS="00787" heldout_run $WC/tmp/heldout_loss/heldout_loss.jsonl 3072 3072 8 "${CK[@]}"
echo FP8_KEEP_EVAL_DONE
