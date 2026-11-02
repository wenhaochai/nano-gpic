#!/bin/bash
#SBATCH -J gpic-abl-smoke
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#FETCHABLE
# Smoke of the optimizer ablation on 8 GPUs (frozen worktree abl_run): AdamW 300 steps; Muon 200 steps, then resumed
# from last.ckpt to 300 (optimizer state save/load); split Muon 300 steps. Validation loss every 100 steps.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
J=/scratch/gpfs/GROUP/USER/project/gpic-wt/abl_run/jobs/train_arm.sh
M=configs_h100/lite_ablation_muon.yaml; S=configs_h100/lite_ablation_smoke.yaml
log() { echo "[$(date +%T)] $*"; }
FAIL=0; step() { local name=$1; shift; log START $name; "$@"; local rc=$?; log "END $name rc=$rc"; [ $rc -eq 0 ] || FAIL=1; }
SID=$(date +%m%d%H%M)   # unique W&B ids per smoke (review N6)
step smoke_adamw bash $J smoke_adamw_$SID --config $S
step smoke_muon_200 bash $J smoke_muon_$SID --config $M --config $S --trainer.max_steps 200
step smoke_muon_resume bash $J smoke_muon_$SID --config $M --config $S
step smoke_cmuon bash $J smoke_cmuon_$SID --config $M --config $S --model.optimizer.init_args.split true
python3 -c "import torch; c=torch.load('$WC/runs/gpic-speed/exp_abl_smoke_muon_$SID/last.ckpt', map_location='cpu', weights_only=False); print('resumed last.ckpt step', c['global_step'], 'keys', sorted(c))" || FAIL=1
log "DONE fail=$FAIL"; exit $FAIL
