#!/bin/bash
#SBATCH -J gpic-ar-overfit
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --time=00:55:00
#SBATCH --mem=200G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Overfit test of the autoregressive model (tools/ar/overfit_ar.py): 4 arms in one 4-GPU job, 24000 steps each: N = 1, N = 32, N = 256 on two seeds. A one-sided check: a failure stops this design, a pass proves little.
# Rule, set before the run, at the last step. The references are the point model's numbers in w1095 at 24000 steps, the same images and metrics.
# 1. N = 1: nn_rms under 0.05.
# 2. N = 32: covered >= 0.6745 (the point model's 0.8125 minus 2 SE).
# 3. N = 256: the mean covered of the two seeds >= 0.7594 (the point model's 0.8086 minus 2 SE).
# Reported with no threshold: near, distinct, the sample time, and tf_rms against nn_rms, which shows whether errors grow along the chain.
# Expected: about 30 min. Code: the frozen worktree gpic-wt/ar_run.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${LADDER_WT:-$WC/project/gpic-wt/ar_run}
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
OUT=$WC/runs/gpic-speed/ar_overfit_$(git -C $WT rev-parse --short=7 HEAD)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD WANDB_MODE=disabled TORCHINDUCTOR_COMPILE_THREADS=8
IFS=, read -r -a DEV <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"   # a guest on part of a node sees its own GPUs here
(( ${#DEV[@]} >= 4 )) || { echo "${#DEV[@]} GPUs"; exit 2; }
g=0
for arm in "1 0" "32 0" "256 0" "256 1"; do
  set -- $arm
  CUDA_VISIBLE_DEVICES=${DEV[$g]} OMP_NUM_THREADS=4 timeout -k 60 3000 python $WT/tools/ar/overfit_ar.py $1 $OUT 24000 $2 > $OUT/n$1_s$2.log 2>&1 &
  g=$((g + 1))
done
fail=0
for p in $(jobs -p); do wait $p || fail=1; done
for arm in "1 0" "32 0" "256 0" "256 1"; do set -- $arm; tail -qn 2 $OUT/n$1_s$2.log | cut -c1-320; done
echo "overfit done fail=$fail $OUT"
exit $fail
