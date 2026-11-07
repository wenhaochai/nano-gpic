#!/bin/bash
#SBATCH -J gpic-ar-diagnose
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --time=00:50:00
#SBATCH --mem=200G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Why does the autoregressive model fail at N = 256 (w1132: covered 0.033, threshold 0.7594)? 4 arms, all N = 256 and 24000 steps, with the same checks: the first design with rescue tests; noise on the context at std 0.1; MAR's random order and bidirectional attention on two seeds. The random arms sample faster (64 backbone calls per image against 256), so their GPUs may idle for about 5 min at the end (an estimate).
# Reading and rule, set before the run: docs/autoregressive.md, section 诊断. The threshold does not change. Single seed: this decides the next design, and no arm is a result on its own.
# Expected: about 25 min. Code: the frozen worktree gpic-wt/ar_run.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${LADDER_WT:-$WC/project/gpic-wt/ar_run}
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
# old code would ignore the options and write one file 4 times (review 2026-11-04)
grep -q -- '--order' $WT/tools/ar/overfit_ar.py && grep -q '_sample_random' $WT/baselines/PixelGen/src/models/transformer/AR_JiT.py && grep -q 'min_mask' $WT/baselines/PixelGen/src/diffusion/flow_matching/training_ar.py || { echo "$WT holds old code"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
OUT=$WC/runs/gpic-speed/ar_diagnose_$(git -C $WT rev-parse --short=7 HEAD)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD WANDB_MODE=disabled TORCHINDUCTOR_COMPILE_THREADS=8
IFS=, read -r -a DEV <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"   # a guest on part of a node sees its own GPUs here
(( ${#DEV[@]} >= 4 )) || { echo "${#DEV[@]} GPUs"; exit 2; }
run() { local g=$1 tag=$2 seed=$3; shift 3; CUDA_VISIBLE_DEVICES=${DEV[$g]} OMP_NUM_THREADS=4 timeout -k 60 2700 python $WT/tools/ar/overfit_ar.py 256 $OUT 24000 $seed --name $tag "$@" > $OUT/$tag.log 2>&1 & }
run 0 base 2
run 1 noise01 2 --context-noise 0.1
run 2 random 0 --order random
run 3 random_s1 1 --order random
fail=0
for p in $(jobs -p); do wait $p || fail=1; done
for tag in base noise01 random random_s1; do tail -qn 3 $OUT/$tag.log | cut -c1-500; done
echo "diagnose done fail=$fail $OUT"
exit $fail
