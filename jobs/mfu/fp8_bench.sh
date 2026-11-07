#!/bin/bash
#SBATCH -J gpic-mfu-fp8bench
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:20:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Native FP8's ceiling per GEMM (tools/mfu/fp8_gemm_bench.py), one width per GPU, all eight at once (no idle GPU).
# usage: fp8_bench.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/fp8bench_$(date +%m%d_%H%M); mkdir -p $OUT
i=0
for C in 640 768 896 1024 1280 1536 1792 2048; do
  CUDA_VISIBLE_DEVICES=$i python $WT/tools/mfu/fp8_gemm_bench.py $C > $OUT/c$C.txt 2>&1 &
  i=$((i + 1))
done
wait
cat $OUT/c*.txt | grep "block total\|forward + dgrad\|Error" | sort -t= -k2 -n | tee $OUT/summary.txt
echo "DONE $OUT"
