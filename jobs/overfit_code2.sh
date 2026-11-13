#!/bin/bash
#SBATCH -J gpic-code-overfit2
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --time=00:40:00
#SBATCH --mem=200G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Retest after w1095 (owner's popup, 2026-11-04): at N = 256, R = 16 fell 2.4 images under the threshold, and the code arms have a lower training loss and more blended samples than the point model. Why?
# 4 arms at N = 256, 24000 steps, one GPU each, on other seeds: point, R = 16, R = 64 with seed 1, R = 16 with seed 2. tools/dist_step/overfit_code.py also samples with z from the encoder.
# Reading, set before the run:
# 1. covered of the same arm over seeds shows the noise of w1095's comparison.
# 2. If enc_near is clearly above near (by more than 0.1) and enc_match is high (above 0.8), the decoder works and the flow makes the blends. Then the flow is the part to improve, and it costs almost nothing to enlarge. z_fit_flow above z_fit_enc says the same in z space.
# 3. If enc_near is not above near, the decoder makes the blends.
# Expected: about 15 min. Code: gpic-wt/code_run.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${LADDER_WT:-$WC/project/gpic-wt/code_run}
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
OUT=$WC/runs/gpic-speed/code_overfit2_$(git -C $WT rev-parse --short=7 HEAD)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD WANDB_MODE=disabled GPIC_CODE_CHECK=4000 TORCHINDUCTOR_COMPILE_THREADS=8
IFS=, read -r -a DEV <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"   # a guest on part of a node sees its own GPUs here
(( ${#DEV[@]} >= 4 )) || { echo "${#DEV[@]} GPUs"; exit 2; }
g=0
for arm in "0 1" "16 1" "64 1" "16 2"; do
  set -- $arm
  CUDA_VISIBLE_DEVICES=${DEV[$g]} OMP_NUM_THREADS=4 timeout -k 60 2000 python $WT/tools/dist_step/overfit_code.py 256 $1 $OUT 24000 $2 > $OUT/n256_r$1_s$2.log 2>&1 &
  g=$((g + 1))
done
fail=0
for p in $(jobs -p); do wait $p || fail=1; done
for arm in "0 1" "16 1" "64 1" "16 2"; do set -- $arm; tail -qn 2 $OUT/n256_r$1_s$2.log | cut -c1-400; done
echo "overfit2 done fail=$fail $OUT"
exit $fail
