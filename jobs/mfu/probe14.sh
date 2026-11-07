#!/bin/bash
#SBATCH -J gpic-mfu-probe14
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:25:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU lever H (docs/mfu.md): launch configurations of the fused clip + AdamW + EMA kernel at d1024's parameter shapes
# (probe12: BLOCK 4096, 8 warps, scalar loads reached about 2.1 TB/s against torch's 2.4). One GPU per task, two
# configurations each (tools/mfu/tests/bench_fused_kernel.py: bit for bit against torch, then the time).
# usage: probe14.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/probe14_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PIXGEN_SRC=$WT/baselines/PixelGen
source $WT/jobs/mfu/pool.sh
cd $WT
B="timeout 900 python tools/mfu/tests/bench_fused_kernel.py --d 1024 --configs"
k=0
for c in 4096:8:0,4096:8:1 1024:4:1,1024:4:0 2048:4:1,2048:8:1 4096:4:1,4096:16:1 8192:8:1,8192:16:1 16384:16:1,512:4:1 1024:2:1,2048:2:1 1024:8:1,2048:16:1; do
  echo "$B $c > $OUT/cfg_$k.log 2>&1"; k=$((k + 1))
done > $OUT/tasks.txt
pool $OUT/tasks.txt $OUT
grep -h "torch:\|BLOCK\|Error\|error" $OUT/cfg_*.log | sort | uniq
echo "DONE $OUT"
