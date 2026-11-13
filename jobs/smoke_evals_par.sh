#!/bin/bash
#SBATCH -J gpic-evals-par-smoke
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:45:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Smoke test of jobs/evals_gmflow.sh --par (one evaluation per GPU, 8 side by side): w1263's 8 evaluations of the Lite d640 run
# (Euler 8, 50 and 4 calls at ms-step 133600-534400, official noise) again in their own root; every FD must match w1263's
# 8-GPU result within 0.5% (the samples are the same: one noise for all prompts; only the batch shapes differ).
set -eo pipefail
WC=/scratch/gpfs/GROUP/USER
R=$WC/runs/gpic-speed/evalpar_smoke
rm -rf $R
t0=$(date +%s)
bash $WC/project/gpic-wt/eval_par_run/jobs/evals_gmflow.sh 0 d640 640 9 2 $R --link $WC/runs/gpic-speed/exp_ladder_d640 --wt eval_par_run --par \
  133600:euler:8 267200:euler:8 400800:euler:8 534400:euler:8 267200:euler:50 534400:euler:50 267200:euler:4 534400:euler:4
echo "[smoke] 8 evaluations in $(( $(date +%s) - t0 )) s"
python - $R/exp_ladder_d640 $WC/runs/gpic-speed/ladder_point_d640/exp_ladder_d640 <<'PY'
import glob, json, os, sys
new, ref = sys.argv[1], sys.argv[2]
worst = 0
for f in sorted(glob.glob(new + "/eval_*/metrics.json")):
    d = os.path.basename(os.path.dirname(f)); a = json.load(open(f)); b = json.load(open(os.path.join(ref, d, "metrics.json")))
    r = abs(a["fd"] / b["fd"] - 1); worst = max(worst, r)
    print(f"[smoke] {d}: fd {a['fd']:.2f} (one GPU) against {b['fd']:.2f} (8 GPUs), {100 * r:.3f}%; world {a['world']}")
assert len(glob.glob(new + "/eval_*/metrics.json")) == 8 and worst < 5e-3, worst
print("[smoke] PASS")
PY
