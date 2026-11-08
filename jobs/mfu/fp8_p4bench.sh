#!/bin/bash
#SBATCH -J gpic-mfu-p4bench
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:30:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Why FP8 phase 4 is slower (w1952): tools/mfu/fp8_phase4_bench.py on 8 GPUs at once, one width and micro-batch per GPU.
# usage: fp8_p4bench.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/p4bench_$(date +%m%d_%H%M); mkdir -p $OUT
cd $WT
i=0
for spec in "1024 32768" "1280 32768" "1536 32768" "1792 16384" "1024 16384" "1280 16384" "1536 16384" "1792 32768"; do
  set -- $spec
  CUDA_VISIBLE_DEVICES=$i python tools/mfu/fp8_phase4_bench.py $1 $2 > $OUT/d$1_m$2.txt 2>&1 &
  i=$((i + 1))
done
wait
for f in $OUT/d*.txt; do grep -v -i warn $f | tail -24; done | tee $OUT/summary.txt
echo "DONE $OUT"
