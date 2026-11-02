#!/bin/bash
#SBATCH -J gpic-abl
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=02:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# One arm of the optimizer ablation: the official GPIC PixelGen config on GPIC Lite with the exact-math speedups
# (C1-C7), 20k steps on 8 GPUs, fixed validation loss every 1k steps, checkpoints ms-010000 and ms-020000 for FD.
# Code: the frozen worktree gpic-wt/abl_run (the reviewed commit), so later edits elsewhere cannot reach a running arm.
# usage: train_arm.sh <arm> [main.py arguments...]   e.g. train_arm.sh muon_lr1e-4 --config configs_h100/lite_ablation_muon.yaml
# Resumable: resumes from <run>/last.ckpt (every 2.5k steps); exits 0 only when 20k steps are done.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
ARM=$1; shift
WT=/scratch/gpfs/GROUP/USER/project/gpic-wt/abl_run
RUN=$WC/runs/gpic-speed/exp_abl_$ARM
mkdir -p $RUN
git -C $WT rev-parse HEAD > $RUN/commit.txt
CKPT_ARGS=()
if [ -f "$RUN/last.ckpt" ]; then CKPT_ARGS=(--ckpt_path "$RUN/last.ckpt"); echo "resuming from $RUN/last.ckpt"; fi
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD
python main.py fit --config configs_h100/lite_ablation.yaml "$@" --exp abl_$ARM \
  --trainer.logger.init_args.name abl_$ARM --trainer.logger.init_args.id abl-$ARM "${CKPT_ARGS[@]}"
rc=$?
echo "main.py exit $rc"
exit $rc
