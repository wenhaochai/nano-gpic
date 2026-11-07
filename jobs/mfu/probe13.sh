#!/bin/bash
#SBATCH -J gpic-mfu-probe13
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:10:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU lever E (docs/mfu.md): where packing and rebalancing lose what they save. The rank imbalance from long-caption
# batches is about 7 ms of a d1024 step and 8 ms of a d1280 step (w1315 traces, tools/mfu/rank_anatomy.py), yet E costs
# 2-3 ms net (w1394, w1397).
# 1. One GPU per task (step_bench, FA3, no exchange): padded and packed at d1280 and d1024, each with a profiler trace of
#    3 steps: the packed path's own kernels and host time.
# 2. Production path (jobs/mfu/widths.sh, 8 GPUs, the same batches): d1280 A+B+C+D and A+B+C+D+E, d1024 A+C+D and
#    A+C+D+E, each with every rank's profiler trace over steps 500-505 and per-stage times (P): the imbalance, the exposed
#    communication and the host plan's cost, with and without E.
# usage: probe13.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/probe13_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TORCHINDUCTOR_CPP_WRAPPER=1 PIXGEN_SRC=$WT/baselines/PixelGen GPIC_MFU_ATTN=fa3
source $WT/jobs/mfu/pool.sh
cd $WT
SB="timeout 1500 python $WT/tools/mfu/step_bench.py --steps 40"
{
  for d in 1280 1024; do
    echo "$SB --d $d --profile 30:33 --out $OUT/pad_d$d.json > $OUT/pad_d$d.log 2>&1"
    echo "$SB --d $d --packed --profile 30:33 --out $OUT/packed_d$d.json > $OUT/packed_d$d.log 2>&1"
  done
} > $OUT/tasks.txt
pool $OUT/tasks.txt $OUT
grep -h "MFU" $OUT/*.log
for f in $OUT/*_trace_rank0.json; do python $WT/tools/mfu/trace_breakdown.py $f --top 30 > ${f%_trace_rank0.json}_breakdown.txt 2>&1; done
bash $WT/jobs/mfu/widths.sh $WT 1024 1280:ABCDP,ABCDEP 1024:ACDP,ACDEP
echo "DONE $OUT"
