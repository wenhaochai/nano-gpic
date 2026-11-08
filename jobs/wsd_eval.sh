#!/bin/bash
#SBATCH -J gpic-wsd-eval
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:30:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# WSD cooldown branches without an EMA (owner 2026-11-07: 「EMA是一定要去掉的」): FD-DINOv2 of a branch's raw weights at its last step.
# make_raw_view.py writes runs/gpic-speed/raw_view/muon_wsd/<tag>/exp_ladder_<size>/ms-step=<step>.ckpt (ema slot = raw weights), then the
# official evaluation (Euler 50, official noise, CFG 1, 50K test captions), one evaluation per GPU.
# usage: wsd_eval.sh <tag> <size> <d> <nb> <nt> <step> [<step> ...]
export NCCL_NVLS_ENABLE=0   # NCCL 2.26.2 NVLS all-gather can permute rank blocks (NVIDIA/nccl #1906; gpic-28 w2090)
set -euo pipefail
WC=/scratch/gpfs/GROUP/USER
TAG=$1 SIZE=$2 WD=$3 NB=$4 NT=$5; shift 5
export TMPDIR=$WC/tmp/job/${FETCHED_JOB_ID:-$SLURM_JOB_ID}; mkdir -p $TMPDIR
for s in "$@"; do RAW_SRC=muon_wsd/$TAG $WC/envs/pixgen/bin/python -I $WC/tmp/ladder_prof/make_raw_view.py $SIZE $s; done
T=(); for s in "$@"; do T+=("$s:euler:50"); done
exec bash $WC/project/gpic-wt/eval_official_run2/jobs/evals_gmflow.sh 0 $SIZE $WD $NB $NT $WC/runs/gpic-speed/raw_view/muon_wsd/$TAG --wt eval_official_run2 "${T[@]}"
