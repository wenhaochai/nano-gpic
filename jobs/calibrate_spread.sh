#!/bin/bash
#SBATCH -J gpic-spread-cal
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:45:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The spread table of a point model (docs/gmflow_hn.md, H1): w1147's EMA at C0/16 and C0/8 (runs/gpic-speed/point_nodrop_d640, ms-step=033408 and 066816), its own
# x-prediction error s_x(t) = sqrt(E||x - x_hat||^2 / D) on 4096 held-out val images (gpic_val_00000.tar) at 64 t in [0.01, 0.99], one process per GPU on disjoint
# images, then the mean. No training, no FD. Expected: about 10 min. Output: runs/gpic-speed/point_spread_table/<step>/table.json (the input of the sampler gm_ode_cal).
# Code: the frozen worktree gpic-wt/gmflow_hn_run; the job records its commit and refuses a worktree with local changes.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/gmflow_hn_run
[ -f "$WT/tools/gmflow/calibrate_spread.py" ] || { echo "$WT holds old code"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
SRC=$WC/runs/gpic-speed/point_nodrop_d640/exp_ladder_d640
ROOT=$WC/runs/gpic-speed/point_spread_table
mkdir -p $ROOT
git -C $WT rev-parse HEAD > $ROOT/commit.txt
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD WANDB_MODE=disabled
IFS=, read -r -a DEV <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
[ ${#DEV[@]} -eq 8 ] || { echo "needs 8 GPUs, got ${CUDA_VISIBLE_DEVICES:-}"; exit 2; }
fail=0
for STEP in 033408 066816; do
  OUT=$ROOT/$STEP
  if [ -f $OUT/table.json ]; then echo "table $STEP done before"; continue; fi
  CKPT=$SRC/ms-step=$STEP.ckpt
  [ -f $CKPT ] || { echo "no $CKPT"; exit 2; }
  mkdir -p $OUT
  rm -f $OUT/shard*.json
  pids=()
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=${DEV[$i]} OMP_NUM_THREADS=8 timeout -k 60 1200 python $WT/tools/gmflow/calibrate_spread.py run $CKPT $OUT --shard $i --shards 8 \
      --images 4096 > $OUT/log$i.txt 2>&1 &
    pids+=($!)
  done
  f=0; for p in "${pids[@]}"; do wait $p || f=1; done
  tail -qn 1 $OUT/log*.txt
  if [ $f -eq 0 ] && python $WT/tools/gmflow/calibrate_spread.py merge $OUT $OUT/table.json; then echo "table $STEP ok"; else echo "table $STEP failed"; fail=1; fi
done
exit $fail
