#!/bin/bash
#SBATCH -J gpic-ar-saved
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --time=00:30:00
#SBATCH --mem=200G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Sampler settings on the saved weights of w1133's MAR-order arm (N = 256, seed 0), no training: 64 and 256 steps, temperature 1 and 0.85. Why: the samples look like one training image with a few stray patches (tools/ar/eval_saved_ar.py).
# Reading, set before the run, in docs/autoregressive.md, section 采样设置. Expected: about 12 min. Code: the frozen worktree gpic-wt/ar_run.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${LADDER_WT:-$WC/project/gpic-wt/ar_run}
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
grep -q 'temperature' $WT/baselines/PixelGen/src/models/transformer/AR_JiT.py && [ -f $WT/tools/ar/eval_saved_ar.py ] || { echo "$WT holds old code"; exit 2; }
SRC=$WC/runs/gpic-speed/ar_diagnose_cabc237
OUT=$WC/runs/gpic-speed/ar_saved_$(git -C $WT rev-parse --short=7 HEAD)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD WANDB_MODE=disabled TORCHINDUCTOR_COMPILE_THREADS=8
IFS=, read -r -a DEV <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
(( ${#DEV[@]} >= 4 )) || { echo "${#DEV[@]} GPUs"; exit 2; }
g=0
for arm in "64 1.0" "256 1.0" "64 0.85" "256 0.85"; do
  set -- $arm
  CUDA_VISIBLE_DEVICES=${DEV[$g]} OMP_NUM_THREADS=4 timeout -k 60 1500 python $WT/tools/ar/eval_saved_ar.py $SRC/random.pt 256 $OUT/s$1_t$2.json $1 $2 > $OUT/s$1_t$2.log 2>&1 &
  g=$((g + 1))
done
fail=0
for p in $(jobs -p); do wait $p || fail=1; done
for arm in "64 1.0" "256 1.0" "64 0.85" "256 0.85"; do set -- $arm; tail -qn 1 $OUT/s$1_t$2.log | cut -c1-400; done
echo "saved eval done fail=$fail $OUT"
exit $fail
