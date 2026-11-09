#!/bin/bash
#SBATCH -J gpic-smoke-eval-run5
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:40:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Smoke of eval_official_run5 (gpic-28 2026-11-09; owner popup 23:5x: 8 GPUs x 0.4 h): (1) evals_group.sh on three runs with run4 results
# (d9 x-d9-c1-lr3p40e-3 133600, d10 x-d10-c4-lr2p72e-2 21248, d9 x-d9-c4-lr6p80e-3 33408), their checkpoints linked under
# runs/gpic-speed/smoke/run5_<time>/; (2) evals_gmflow.sh with two tasks on one root (the two d9 checkpoints as steps 1000 and 2000), so
# its own pipeline runs; then each score beside its run4 value, and each part's wall time beside the run4 serial sum. GPU use is
# sampled every 2 s (idle gaps, memory with a predict and a scoring on the same GPUs). The raw views it wrote are removed at the end.
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
source $WC/project/gpic/jobs/env.sh
export TORCH_LOGS=recompiles
TAGN=run5_$(date +%m%d_%H%M)
RV=$WC/runs/gpic-speed/raw_view
S=$WC/runs/gpic-speed/smoke/$TAGN
link() {   # <name> <run under raw_view> <size> <step> [<linked as step>]
  local d=$S/$1/exp_ladder_$3; mkdir -p $d
  ln -sf $RV/$2/exp_ladder_$3/ms-step=$(printf %06d $4).ckpt $d/ms-step=$(printf %06d ${5:-$4}).ckpt
}
link A muon_sweep/x-d9-c1-lr3p40e-3 d640 133600
link C muon_sweep/x-d10-c4-lr2p72e-2 d768 21248
link B muon_sweep/x-d9-c4-lr6p80e-3 d640 33408
link same muon_sweep/x-d9-c1-lr3p40e-3 d640 133600 1000
link same muon_sweep/x-d9-c4-lr6p80e-3 d640 33408 2000
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -lms 2000 > $S/gpu_use.csv 2>/dev/null &
SMI=$!
W5=$WC/project/gpic-wt/eval_official_run5/jobs
t0=$(date +%s)
bash $W5/evals_group.sh --wt eval_official_run5 smoke/$TAGN/A:d640:640:9:2:133600 smoke/$TAGN/C:d768:768:10:2:21248 smoke/$TAGN/B:d640:640:9:2:33408
echo "group rc=$?"
t1=$(date +%s)
bash $W5/evals_gmflow.sh 0 d640 640 9 2 $S/same --wt eval_official_run5 1000:euler:50 2000:euler:50
echo "evals_gmflow pipeline rc=$?"
t2=$(date +%s)
kill $SMI 2>/dev/null
source $WC/envs/pixgen/bin/activate
python - $S $RV/smoke/$TAGN $RV $WC/runs/gpic-speed/eval_smoke/run4_1108_2332 $((t1 - t0)) $((t2 - t1)) <<'PY'
import csv, json, os, sys
S, G, RV, R4, wg, ws = sys.argv[1:4] + [sys.argv[4]] + list(map(int, sys.argv[5:7]))
rd = lambda d, f: json.load(open(f"{d}/{f}")) if os.path.exists(f"{d}/{f}") else {}
ev = lambda st: f"eval_{st:06d}_euler_calls50_official"
ref = {"A": (f"{RV}/muon_sweep/x-d9-c1-lr3p40e-3/exp_ladder_d640/{ev(133600)}", 133600, "d640"),
       "C": (f"{RV}/muon_sweep/x-d10-c4-lr2p72e-2/exp_ladder_d768/{ev(21248)}", 21248, "d768"),
       "B": (f"{R4}/x-d9-c4/exp_ladder_d640/{ev(33408)}", 33408, "d640")}
def show(label, new_d, old_d):
    a, b = rd(new_d, "metrics.json"), rd(old_d, "metrics.json")
    ai, bi = rd(new_d, "metrics_inception.json"), rd(old_d, "metrics_inception.json")
    p, q = rd(new_d, "phase_times.json"), rd(old_d, "phase_times.json")
    serial = q.get("predict_s", 0) + max(q.get("eval_s", 0), q.get("inception_s", 0))
    print(f"{label}: FD {a.get('fd', float('nan')):.3f} (run4 {b.get('fd', float('nan')):.3f}), FID {ai.get('inception_fd', float('nan')):.3f} "
          f"(run4 {bi.get('inception_fd', float('nan')):.3f}); phases {p} (run4 {q}; run4 serial {serial} s)")
    return serial
tot = 0
for k in ("A", "C", "B"):
    old, st, size = ref[k]
    tot += show(f"group {k}", f"{G}/{k}/exp_ladder_{size}/{ev(st)}", old)
print(f"group wall {wg} s; run4 serial sum {tot} s (raw views not counted)")
tot = 0
for st, k in ((1000, "A"), (2000, "B")):
    tot += show(f"same-root step {st} ({k})", f"{S}/same/exp_ladder_d640/{ev(st)}", ref[k][0])
print(f"same-root wall {ws} s; run4 serial sum {tot} s")
rows = [r for r in csv.reader(open(f"{S}/gpu_use.csv")) if len(r) == 4]
if rows:
    util = [float(r[2].strip().split()[0]) for r in rows]; mem = [float(r[3].strip().split()[0]) for r in rows]
    n = len(util)
    print(f"GPU samples {n}: share at 0% {sum(u == 0 for u in util) / n:.3f}, mean util {sum(util) / n:.1f}%, peak memory {max(mem):.0f} MiB")
PY
rm -f $RV/smoke/$TAGN/*/exp_ladder_*/ms-step=*.ckpt   # the smoke's own raw views (0.8-1.3 GB each); the scores stay
echo EVAL_SMOKE_RUN5_DONE $S
