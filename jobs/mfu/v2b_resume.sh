#!/bin/bash
#SBATCH -J gpic-mfu-v2b
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:45:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU freeze checks (a) and (b) again after the FSDP2 resume fix (genmodel, 2026-11-06): jobs/smoke_full.sh's resume part
# (d640 stop at 1000 / resume to 4800, tools/check_shard_resume.py) twice: with the head's defaults (d640: A, C, D, F) and
# with FSDP2 forced on (GPIC_MFU_FSDP=1: the path the d1280-d1792 runs take); then the layout of the new ms and last
# checkpoints against 5cd3372's smoke, and gmflow_run4's real evaluation path on the FSDP2 run's ms-step=001600.
# usage: v2b_resume.sh <code worktree at a clean commit>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/v2b_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PIXGEN_SRC=$WT/baselines/PixelGen
source $WT/jobs/mfu/gpu_checks.sh
gpu_checks $WT $OUT || { echo "GPU checks failed"; exit 3; }
REF=$WC/runs/gpic-speed/full_smoke_1106_1433/resume/exp_ladder_d640
EWT=$WC/project/gpic-wt/gmflow_run4
for mode in default fsdp; do
  [ $mode = fsdp ] && export GPIC_MFU_FSDP=1
  bash $WT/jobs/smoke_full.sh $WT resume > $OUT/smoke_$mode.log 2>&1
  rc=$?; echo "[$(date +%T)] smoke resume ($mode) rc=$rc" | tee -a $OUT/summary.txt
  SMK=$(ls -dt $WC/runs/gpic-speed/full_smoke_* | head -1)
  grep -h " ok\| FAIL\|CHECKS" $SMK/phases.log | tee -a $OUT/summary.txt
  N=$SMK/resume/exp_ladder_d640
  ( cd $WT/baselines/PixelGen && PYTHONPATH=$PWD python $WT/tools/mfu/compare_ckpt_layout.py $REF/ms-step=001600.ckpt $N/ms-step=001600.ckpt \
    $REF/ms-step=004800.ckpt $N/ms-step=004800.ckpt $REF/last.ckpt $N/last.ckpt ) 2>&1 | tee -a $OUT/summary.txt
  if [ $mode = fsdp ]; then
    EV=$OUT/eval
    mkdir -p $EV/exp_ladder_d640 && ln -sf $N/ms-step=001600.ckpt $EV/exp_ladder_d640/
    GMFLOW_WT=$EWT WANDB_MODE=disabled EVAL_LIMIT_BATCHES=1 LADDER_RUN_ROOT=$EV timeout -k 60 900 bash $EWT/jobs/eval_gmflow.sh 0 1600 euler 8 > $OUT/eval.log 2>&1
    rc=$?
    chk=$(cat $EV/exp_ladder_d640/eval_001600_euler_calls8*/sample_check.txt 2>/dev/null)
    echo "[$(date +%T)] eval path rc=$rc sample_check: $chk" | tee -a $OUT/summary.txt
  fi
done
echo "DONE $OUT"
