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
for a in smoke_adamw smoke_muon smoke_cmuon; do d=$WC/runs/gpic-speed/exp_abl_$a; case "$d" in */exp_abl_smoke_*) rm -rf -- "$d";; esac; done
log() { echo "[$(date +%T)] $*"; }
log START smoke_adamw; bash $J smoke_adamw --config $S; log "END smoke_adamw rc=$?"
log START smoke_muon_200; bash $J smoke_muon --config $M --config $S --trainer.max_steps 200; log "END smoke_muon_200 rc=$?"
log START smoke_muon_resume; bash $J smoke_muon --config $M --config $S; log "END smoke_muon_resume rc=$?"
log START smoke_cmuon; bash $J smoke_cmuon --config $M --config $S --model.optimizer.init_args.split true; log "END smoke_cmuon rc=$?"
log DONE
