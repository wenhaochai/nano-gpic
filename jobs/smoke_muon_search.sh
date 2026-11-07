#!/bin/bash
#SBATCH -J gpic-muon-search-smoke
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:30:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Smoke test of jobs/train_muon_search.sh: d640, 64 optimizer steps of 1024 images with the middle Muon lr, in its own root
# (runs/gpic-speed/muon_search_smoke); checks the merged config (Muon lrs, batch 1024, EMA 0.9996, warmup), the ms checkpoints
# at 16/32/48/64 and that the evaluation code loads the Muon checkpoint (2 prediction batches per rank, no FD).
set -eo pipefail
WC=/scratch/gpfs/GROUP/USER
WT=$WC/project/gpic-wt/muon_search_run
R=$WC/runs/gpic-speed/muon_search_smoke
rm -rf $R
SEARCH_ROOT=$R bash $WT/jobs/train_muon_search.sh d640 640 9 2 64 6.4e-4 1.7e-3 smoke
ls $R/exp_ladder_d640/ms-step=000064*.ckpt
python - $R/exp_ladder_d640 <<'PY'
import glob, sys, yaml
c = yaml.safe_load(open(sorted(glob.glob(sys.argv[1] + "/config-*.yaml"))[-1]))
o = c["model"]["optimizer"]["init_args"]; t = c["trainer"]; e = c["model"]["ema_tracker"]["init_args"]
got = (o["lr"], o["muon_lr"], o["update_scale"], o["split"], t["accumulate_grad_batches"], e["decay"], c["model"]["lr_scheduler"]["init_args"]["num_warmup_steps"])
print("[smoke] adamw lr, muon lr, scale, split, accumulation, ema, warmup:", got)
assert got == (6.4e-4, 1.7e-3, "mup", True, 4, 0.9996, 1000), got
PY
source $WC/project/gpic/jobs/env.sh
export GMFLOW_WT=$WC/project/gpic-wt/gmflow_run4 LADDER_RUN_ROOT=$R EVAL_SIZE=d640 EVAL_D=640 EVAL_NB=9 EVAL_NT=2 EVAL_NOISE=official EVAL_LIMIT_BATCHES=2
bash $GMFLOW_WT/jobs/eval_gmflow.sh 0 64 euler 8
echo "[smoke] PASS"
