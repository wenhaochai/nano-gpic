#!/bin/bash
#SBATCH -J gpic-ar-long
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --time=01:40:00
#SBATCH --mem=200G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Test B: the MAR-order model on 256 training images to 48000 steps, 4 arms, each followed by the redraw and following tests on its weights. Reading, set before the run, in docs/autoregressive.md, section 测试 B. The threshold does not change. Expected: about 1 h. Code: the frozen worktree gpic-wt/ar_long.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${LADDER_WT:-$WC/project/gpic-wt/ar_long}
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
grep -q -- '--few-known' $WT/tools/ar/overfit_ar.py && grep -q 'HEAD_DEPTH' $WT/tools/ar/refine_ar.py && grep -q 'HEAD_DEPTH' $WT/tools/ar/follow_ar.py || { echo "$WT holds old code"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
OUT=$WC/runs/gpic-speed/ar_long_$(git -C $WT rev-parse --short=7 HEAD)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD WANDB_MODE=disabled TORCHINDUCTOR_COMPILE_THREADS=8
IFS=, read -r -a DEV <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
(( ${#DEV[@]} >= 4 )) || { echo "${#DEV[@]} GPUs"; exit 2; }
arm() {   # gpu tag seed head_depth [overfit options]
  local g=$1 tag=$2 seed=$3 depth=$4; shift 4
  export CUDA_VISIBLE_DEVICES=${DEV[$g]} OMP_NUM_THREADS=4
  timeout -k 60 5000 python $WT/tools/ar/overfit_ar.py 256 $OUT 48000 $seed --order random --head-depth $depth --name $tag "$@" > $OUT/$tag.log 2>&1 || { echo "failed: train $tag"; return 1; }
  timeout -k 60 600 python $WT/tools/ar/refine_ar.py $OUT/$tag.pt 256 $OUT/${tag}_refine.json 11 $depth > $OUT/${tag}_refine.log 2>&1 || { echo "failed: refine $tag"; return 1; }
  timeout -k 60 600 python $WT/tools/ar/follow_ar.py $OUT/$tag.pt 256 $OUT/${tag}_follow.json 11 $depth > $OUT/${tag}_follow.log 2>&1 || { echo "failed: follow $tag"; return 1; }
}
arm 0 long_s0 0 3 &
arm 1 long_s1 1 3 &
arm 2 long_head6 0 6 &
arm 3 long_few 0 3 --few-known 0.5 &
fail=0
for p in $(jobs -p); do wait $p || fail=1; done
echo "long done fail=$fail $OUT"
exit $fail
