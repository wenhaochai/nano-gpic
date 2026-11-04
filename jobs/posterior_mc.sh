#!/bin/bash
#SBATCH -J gpic-posterior-mc
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --time=00:30:00
#SBATCH --mem=200G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The distribution at one step of the real model (tools/dist_step/posterior_mc.py), for the owner's Q&A page: w1053's EMA at C0/8, 128 images of shard 0, 32 samples per (image, t), one GPU per shard. No training. Expected: about 4 min.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/dist_step
OUT=$WC/runs/gpic-speed/posterior_mc_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD WANDB_MODE=disabled
IFS=, read -r -a DEV <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
fail=0
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${DEV[$i]} OMP_NUM_THREADS=4 timeout -k 60 1500 python $WT/tools/dist_step/posterior_mc.py $WC/runs/gpic-speed/ladder_seed_same/exp_ladder_d640/ms-step=066816.ckpt $OUT $i 4 128 32 > $OUT/log$i.txt 2>&1 &
done
for p in $(jobs -p); do wait $p || fail=1; done
tail -qn 1 $OUT/log*.txt
echo "posterior mc done fail=$fail $OUT"
exit $fail
