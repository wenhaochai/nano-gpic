#!/bin/bash
#SBATCH -J gpic-energy-overfit
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --time=00:18:00
#SBATCH --mem=200G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Test E (docs/energy_head.md): the energy head against the point model, overfit on N = 256 real images with the null caption, 24000 steps, one arm per GPU:
# point seed 0 (control; its effective_images was never measured); energy head seed 0 and seed 1; the implicit model in the repo (xi_block = 6, M = 2) seed 0.
# Reading rule, set before the run: docs/energy_head.md, section 读法. The threshold is covered >= 0.7594 on 2 seeds.
# Expected: about 14 min for the slowest arm (implicit), an estimate from w1113's H100 timings and an A40 timing of each arm's train step (point 74.9 ms, energy 82.5, implicit 97.5): no H100 was free to time it. The implicit arm trains 32% more FLOPs per step, so the other GPUs idle about 3 min at the end.
# Code: the worktree gpic-wt/energy_head.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${LADDER_WT:-$WC/project/gpic-wt/energy_head}
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
[ -f $WT/tools/energy/overfit_energy.py ] && grep -q 'class EnergyHead' $WT/baselines/PixelGen/src/models/transformer/JiT_T2I.py && grep -q 'effective_images' $WT/tools/energy/overfit_energy.py || { echo "$WT holds old code"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
OUT=$WC/runs/gpic-speed/energy_overfit_$(git -C $WT rev-parse --short=7 HEAD)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD WANDB_MODE=disabled TORCHINDUCTOR_COMPILE_THREADS=8
IFS=, read -r -a DEV <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"   # a guest on part of a node sees its own GPUs here
(( ${#DEV[@]} >= 4 )) || { echo "${#DEV[@]} GPUs"; exit 2; }
ARMS=("point 0" "energy 0" "energy 1" "implicit 0")
g=0
for arm in "${ARMS[@]}"; do
  set -- $arm
  CUDA_VISIBLE_DEVICES=${DEV[$g]} OMP_NUM_THREADS=4 timeout -k 60 1000 python $WT/tools/energy/overfit_energy.py $1 256 $OUT 24000 $2 > $OUT/n256_$1_s$2.log 2>&1 &
  g=$((g + 1))
done
fail=0
for p in $(jobs -p); do wait $p || fail=1; done
for arm in "${ARMS[@]}"; do set -- $arm; tail -qn 2 $OUT/n256_$1_s$2.log | cut -c1-500; done
echo "energy overfit done fail=$fail $OUT"
exit $fail
