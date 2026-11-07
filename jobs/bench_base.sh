#!/bin/bash
#SBATCH -J gpic-bench-base
#SBATCH --gres=gpu:8
#SBATCH --time=01:30:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#FETCHABLE
# who-ate-my-flops baseline: (1) unprofiled bench, (2) profiled bench (steps 100-103), (3) correctness x3 + kernel-swap.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=/scratch/gpfs/GROUP/USER/project/gpic-wt/perf
WS=$WT/workspace-who-ate-my-flops
SHA=$(git -C $WT rev-parse --short HEAD)
bash $WT/jobs/bench_train.sh $WS/runs/base_${SHA}_r1
bash $WT/jobs/bench_train.sh $WS/runs/base_${SHA}_prof 100:103
mkdir -p $WS/runs/corr_${SHA}
cd $WT/baselines/PixelGen && export PYTHONPATH=$PWD
for i in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((i-1)) python $WT/tools/correctness.py record $WS/runs/corr_${SHA}/base$i.pt > $WS/runs/corr_${SHA}/base$i.log 2>&1 &
  [ $i = 1 ] && wait   # first one writes the fixed batch
done
CUDA_VISIBLE_DEVICES=3 CORR_SDPA=math python $WT/tools/correctness.py record $WS/runs/corr_${SHA}/sdpa_math.pt > $WS/runs/corr_${SHA}/sdpa_math.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 CORR_COMPILE=0 python $WT/tools/correctness.py record $WS/runs/corr_${SHA}/eager.pt > $WS/runs/corr_${SHA}/eager.log 2>&1 &
wait
ls -la $WS/runs/*_${SHA}*
