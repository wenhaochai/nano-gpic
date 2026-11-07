#!/bin/bash
#SBATCH -J gpic-mfu-v2
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:55:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU freeze checks (b) and (c) for the full ladder (genmodel, 2026-11-06), with every lever on (jobs/train_ladder.sh
# defaults): 1. the ladder's own smoke, jobs/smoke_full.sh (d1536 and d1792, 320 steps each with last.ckpt every 100 steps;
# then d640 stop at 1000 / resume to 4800 with the shard log, checked by tools/check_shard_resume.py); 2. the full data set
# (gpic_train_{00000..07999}.tar and its text cache) at d640 for 320 steps; 3. tools/mfu/check_ckpt.py-style checks of the
# smoke's checkpoints: the ms checkpoint's keys and shapes against an ms checkpoint of 5cd3372 at d640 (w1322's smoke).
# usage: v2_smoke.sh <code worktree at a clean commit>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/v2_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PIXGEN_SRC=$WT/baselines/PixelGen
source $WT/jobs/mfu/gpu_checks.sh
gpu_checks $WT $OUT || { echo "GPU checks failed"; exit 3; }
bash $WT/jobs/smoke_full.sh $WT all > $OUT/smoke_full.log 2>&1
rc=$?; echo "[$(date +%T)] smoke_full rc=$rc" | tee -a $OUT/summary.txt
grep -h "DONE\|ok\|FAIL\|fail" $OUT/smoke_full.log | tail -15 | tee -a $OUT/summary.txt
# 3. checkpoint layout against the frozen code's smoke (5cd3372, full_smoke_1106_1433): ms (evaluation) and last (resume)
SMK=$(ls -dt $WC/runs/gpic-speed/full_smoke_* | head -1)
REF=$WC/runs/gpic-speed/full_smoke_1106_1433/resume/exp_ladder_d640
( cd $WT/baselines/PixelGen && PYTHONPATH=$PWD python $WT/tools/mfu/compare_ckpt_layout.py $REF/ms-step=001600.ckpt $SMK/resume/exp_ladder_d640/ms-step=001600.ckpt \
  $REF/ms-step=004800.ckpt $SMK/resume/exp_ladder_d640/ms-step=004800.ckpt $REF/last.ckpt $SMK/resume/exp_ladder_d640/last.ckpt ) 2>&1 | tee -a $OUT/summary.txt
# 4. (a) the real evaluation path (genmodel, 2026-11-06): gmflow_run4's own eval_gmflow.sh on the smoke's ms-step=001600
#    through a scratch root with a link to it, one batch per rank; pass = exit 0 and a sample_check.txt
EV=$OUT/eval
mkdir -p $EV/exp_ladder_d640 && ln -sf $SMK/resume/exp_ladder_d640/ms-step=001600.ckpt $EV/exp_ladder_d640/
EWT=$WC/project/gpic-wt/gmflow_run4
GMFLOW_WT=$EWT WANDB_MODE=disabled EVAL_LIMIT_BATCHES=1 LADDER_RUN_ROOT=$EV timeout -k 60 900 bash $EWT/jobs/eval_gmflow.sh 0 1600 euler 8 > $OUT/eval.log 2>&1
rc=$?
chk=$(cat $EV/exp_ladder_d640/eval_001600_euler_calls8*/sample_check.txt 2>/dev/null)
echo "[$(date +%T)] eval path rc=$rc sample_check: $chk" | tee -a $OUT/summary.txt
# 2. the full data set
LADDER_WT=$WT LADDER_RUN_ROOT=$OUT/full LADDER_PARTIAL_OK=1 BENCH_OUT=$OUT/full/bench GPIC_SHARD_SEED=20261106 \
  bash $WT/jobs/train_ladder.sh d640 640 9 2 6400 --data.train_dataset.init_args.tar_pattern 'gpic_train_{00000..07999}.tar' \
  --trainer.max_steps 320 --trainer.logger.init_args.name mfu_v2_full --trainer.logger.init_args.id mfu-v2-full-$(date +%s) > $OUT/full.log 2>&1
rc=$?; echo "[$(date +%T)] full data rc=$rc" | tee -a $OUT/summary.txt
python $WT/tools/mfu/bench_summary.py $OUT/full/bench --skip 50 | tee -a $OUT/summary.txt
rm -f $OUT/full/exp_ladder_d640/*.ckpt
echo "DONE $OUT"
