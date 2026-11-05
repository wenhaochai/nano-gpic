#!/bin/bash
#SBATCH -J gpic-ar-time
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --time=00:25:00
#SBATCH --mem=200G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Where autoregressive sampling spends its time, and whether the whole head flow in one compiled graph helps (tools/ar/time_head.py), with GPU utilization each second. No training. 4 seeds, one per GPU. Expected: about 8 min. Code: the frozen worktree gpic-wt/ar_time.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${LADDER_WT:-$WC/project/gpic-wt/ar_time}
[ -f "$WT/tools/ar/time_head.py" ] || { echo "$WT holds old code"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
SRC=$WC/runs/gpic-speed/ar_diagnose_cabc237/random.pt
OUT=$WC/runs/gpic-speed/ar_time_$(git -C $WT rev-parse --short=7 HEAD)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD WANDB_MODE=disabled TORCHINDUCTOR_COMPILE_THREADS=8
IFS=, read -r -a DEV <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
(( ${#DEV[@]} >= 4 )) || { echo "${#DEV[@]} GPUs"; exit 2; }
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw --format=csv,noheader,nounits -l 1 > $OUT/gpu_util.csv &
MON=$!
pids=()
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${DEV[$g]} OMP_NUM_THREADS=4 timeout -k 60 1200 python $WT/tools/ar/time_head.py $SRC $OUT/seed$g.json $((30 + g)) > $OUT/seed$g.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
kill $MON 2>/dev/null
for g in 0 1 2 3; do tail -qn 3 $OUT/seed$g.log | cut -c1-400; done
echo "time done fail=$fail $OUT"
exit $fail
