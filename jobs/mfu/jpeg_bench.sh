#!/bin/bash
#SBATCH -J gpic-mfu-jpeg
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:20:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# GPU JPEG decoding against Pillow (tools/mfu/jpeg_gpu_bench.py), one copy per GPU at once (owner 2026-11-07: 可以用GPU解码).
# usage: jpeg_bench.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/jpegbench_$(date +%m%d_%H%M); mkdir -p $OUT
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python $WT/tools/mfu/jpeg_gpu_bench.py $((i * 97)) 128 > $OUT/g$i.txt 2>&1 &
done
wait
cat $OUT/g*.txt | grep "GPU \|Error" | tee $OUT/summary.txt
echo "DONE $OUT"
