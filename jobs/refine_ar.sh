#!/bin/bash
#SBATCH -J gpic-ar-refine
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --time=00:25:00
#SBATCH --mem=200G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Drawing tokens again on w1133's MAR-order weights, no training (tools/ar/refine_ar.py): both weights, two sampling seeds each, one arm per GPU. Reading, set before the run, in docs/autoregressive.md, section 重新抽样. Expected: about 6 min. Code: the frozen worktree gpic-wt/ar_run.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${LADDER_WT:-$WC/project/gpic-wt/ar_run}
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
[ -f $WT/tools/ar/refine_ar.py ] && grep -q 'def redraw' $WT/baselines/PixelGen/src/models/transformer/AR_JiT.py || { echo "$WT holds old code"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
SRC=$WC/runs/gpic-speed/ar_diagnose_cabc237
OUT=$WC/runs/gpic-speed/ar_refine_$(git -C $WT rev-parse --short=7 HEAD)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD WANDB_MODE=disabled TORCHINDUCTOR_COMPILE_THREADS=8
IFS=, read -r -a DEV <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
(( ${#DEV[@]} >= 4 )) || { echo "${#DEV[@]} GPUs"; exit 2; }
g=0
for arm in "random 11" "random 12" "random_s1 11" "random_s1 12"; do
  set -- $arm
  CUDA_VISIBLE_DEVICES=${DEV[$g]} OMP_NUM_THREADS=4 timeout -k 60 1200 python $WT/tools/ar/refine_ar.py $SRC/$1.pt 256 $OUT/$1_seed$2.json $2 > $OUT/$1_seed$2.log 2>&1 &
  g=$((g + 1))
done
fail=0
for p in $(jobs -p); do wait $p || fail=1; done
echo "refine done fail=$fail $OUT"
exit $fail
