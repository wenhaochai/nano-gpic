#!/bin/bash
#SBATCH -J gpic-code-overfit
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:40:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Overfit test of plan A (tools/dist_step/overfit_code.py): 8 arms side by side, one per GPU. N = 1 checks the plumbing; N = 32 and 256 are a one-sided check: a failure stops plan A, a pass proves little.
# Pass, set before the run: at N = 1 both arms return the image (nn_rms under 0.05 at 50 calls at the last step). At N = 32 and 256 a code arm's covered is at least the point arm's at the last step.
# Expected: about 12 min. Code: the frozen worktree gpic-wt/code_run.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${LADDER_WT:-$WC/project/gpic-wt/code_run}
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
TS=$(date +%m%d_%H%M)
OUT=$WC/runs/gpic-speed/code_overfit_$TS
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD WANDB_MODE=disabled GPIC_CODE_CHECK=1000
g=0
for arm in "1 0" "1 16" "32 0" "32 16" "32 64" "256 0" "256 16" "256 64"; do
  set -- $arm
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 timeout -k 60 1800 python $WT/tools/dist_step/overfit_code.py $1 $2 $OUT 8000 > $OUT/n$1_r$2.log 2>&1 &
  g=$((g + 1))
done
fail=0
for p in $(jobs -p); do wait $p || fail=1; done
tail -qn 2 $OUT/n*_r*.log | cut -c1-260
echo "overfit done fail=$fail $OUT"
exit $fail
