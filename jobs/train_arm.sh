#!/bin/bash
#SBATCH -J gpic-abl
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=03:00:00
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
# GPU health gate: the GPU-hours comparison needs healthy GPUs; exit 98 = requeue elsewhere
python $WT/tools/gpu_health.py > $RUN/gpu_health_$(date +%s).txt 2>&1 || { tail -3 $RUN/gpu_health_*.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
CKPT_ARGS=()
LAST=$(ls -t $RUN/last*.ckpt 2>/dev/null | head -1)
if [ -n "$LAST" ]; then CKPT_ARGS=(--ckpt_path "$LAST"); echo "resuming from $LAST"; fi
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD
export GPIC_SHARD_SEED=$(date +%s)   # shard order shuffled per launch, shared by all ranks (review S4)
python main.py fit --config configs_h100/lite_ablation.yaml "$@" --tags.exp abl_$ARM \
  --trainer.logger.init_args.name abl_$ARM --trainer.logger.init_args.id abl-$ARM "${CKPT_ARGS[@]}"
rc=$?
echo "main.py exit $rc"
exit $rc
