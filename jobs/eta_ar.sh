#!/bin/bash
#SBATCH -J gpic-ar-eta
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --time=00:40:00
#SBATCH --mem=200G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Test A: the mask-state jump process with stochasticity eta on w1133's 2 weights, no training (tools/ar/eval_eta.py). 14 runs from one shared list on 4 GPUs. Reading, set before the run, in docs/autoregressive.md, section 测试 A. Expected: about 15 min. Code: the frozen worktree gpic-wt/ar_run.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${LADDER_WT:-$WC/project/gpic-wt/ar_run}
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
[ -f $WT/tools/ar/eval_eta.py ] && grep -q 'def sample_eta' $WT/baselines/PixelGen/src/models/transformer/AR_JiT.py || { echo "$WT holds old code"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
SRC=$WC/runs/gpic-speed/ar_diagnose_cabc237
OUT=$WC/runs/gpic-speed/ar_eta_$(git -C $WT rev-parse --short=7 HEAD)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD WANDB_MODE=disabled TORCHINDUCTOR_COMPILE_THREADS=8
IFS=, read -r -a DEV <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
(( ${#DEV[@]} >= 4 )) || { echo "${#DEV[@]} GPUs"; exit 2; }
TASKS=()
for w in random random_s1; do for e in 0 1 2 5 10 20; do TASKS+=("$w $e 64"); done; TASKS+=("$w 20 256"); done
echo 0 > $OUT/.next
rm -f $OUT/.fail
worker() {
  local gpu=$1 i w e k
  while :; do
    i=$(flock $OUT/.lock bash -c 'n=$(cat "$0"); echo $((n + 1)) > "$0"; echo $n' $OUT/.next)
    (( i < ${#TASKS[@]} )) || break
    read -r w e k <<<"${TASKS[$i]}"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 timeout -k 60 1500 python $WT/tools/ar/eval_eta.py $SRC/$w.pt 256 $OUT/${w}_eta${e}_s$k.json $e 21 --ar-steps $k > $OUT/${w}_eta${e}_s$k.log 2>&1 || { echo "failed: $w eta $e steps $k"; touch $OUT/.fail; }
  done
}
for g in 0 1 2 3; do worker ${DEV[$g]} & done
wait
[ ! -e $OUT/.fail ]; fail=$?
echo "eta done fail=$fail $OUT"
exit $fail
