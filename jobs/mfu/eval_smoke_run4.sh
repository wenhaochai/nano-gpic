#!/bin/bash
#SBATCH -J gpic-smoke-eval-run4
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Smoke of eval_official_run4 (gpic-28 2026-11-08; the eval idle-GPU work, owner: 「一定要处理空闲」): two evaluations whose run2/run3 results
# exist, d9 (x-d9-c4-lr6p80e-3 step 33408: predict 194 s, eval 71 s, Inception 60 s) and d16 (fp8-d20-bf16 step 13600, FD 802.7), each
# into a fresh root through --link; then their metrics and phase times beside the old ones, and the recompiles of each predict.
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
export TORCH_LOGS=recompiles
R=$WC/runs/gpic-speed/eval_smoke/run4_$(date +%m%d_%H%M)
E=$WC/project/gpic-wt/eval_official_run4/jobs/evals_gmflow.sh
RV=$WC/runs/gpic-speed/raw_view
bash $E 0 d640 640 9 2 $R/x-d9-c4 --link $RV/muon_sweep/x-d9-c4-lr6p80e-3/exp_ladder_d640 --wt eval_official_run4 33408:euler:50
bash $E 0 d1536 1536 16 4 $R/fp8-d20-bf16 --link $RV/muon_fp8recipe/fp8-d20-bf16/exp_ladder_d1536 --wt eval_official_run4 13600:euler:50
source $WC/envs/pixgen/bin/activate
python - $R $RV <<'PY'
import json, os, sys
R, RV = sys.argv[1:3]
for name, old, step in (("x-d9-c4", "muon_sweep/x-d9-c4-lr6p80e-3/exp_ladder_d640", 33408), ("fp8-d20-bf16", "muon_fp8recipe/fp8-d20-bf16/exp_ladder_d1536", 13600)):
    size = old.split("exp_ladder_")[1]
    ev = f"eval_{step:06d}_euler_calls50_official"
    new_d, old_d = f"{R}/{name}/exp_ladder_{size}/{ev}", f"{RV}/{old}/{ev}"
    rd = lambda d, f: json.load(open(f"{d}/{f}")) if os.path.exists(f"{d}/{f}") else {}
    for f in ("metrics.json", "metrics_inception.json", "phase_times.json"):
        a, b = rd(new_d, f), rd(old_d, f)
        keys = [k for k in a if isinstance(a[k], (int, float)) and k in b]
        print(f"{name} {f}: " + ", ".join(f"{k} {a[k]:.6g} (old {b[k]:.6g})" for k in keys))
    log = f"{R}/{name}/eval_{step}_euler_50_official.log"
    pl = f"{new_d}/predict.log" if os.path.exists(f"{new_d}/predict.log") else None
    for cand in (f"{R}/{name}/{ev}/predict.log", f"{new_d}/predict.log"):
        if os.path.exists(cand):
            print(f"{name}: recompiles in the predict log: {open(cand, errors='replace').read().count('Recompiling')}")
            break
PY
echo EVAL_SMOKE_RUN4_DONE $R
