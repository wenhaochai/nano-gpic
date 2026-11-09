#!/bin/bash
#SBATCH -J gpic-smoke-eval-run6
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:40:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Smoke of eval_official_run6 (gpic-28 2026-11-09; the eval idle-GPU work): the run5 smoke's five evaluations (three runs through
# evals_group.sh, two tasks on one root through evals_gmflow.sh) plus a d16 run in the group (fp8-d20-bf16 13600: memory with two predicts
# and a scoring on the same GPUs). run6: the next predict sets up and compiles beside the current one and waits at its batch 1; samples
# as .npy; the Inception pass split over the 8 GPUs. Every score must equal run5's (run4's for d16); wall time beside run5's smoke; GPU
# use every 2 s.
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
source $WC/project/gpic/jobs/env.sh
export TORCH_LOGS=recompiles
TAGN=run6_$(date +%m%d_%H%M)
RV=$WC/runs/gpic-speed/raw_view
S=$WC/runs/gpic-speed/smoke/$TAGN
link() {   # <name> <run under raw_view> <size> <step> [<linked as step>]
  local d=$S/$1/exp_ladder_$3; mkdir -p $d
  ln -sf $RV/$2/exp_ladder_$3/ms-step=$(printf %06d $4).ckpt $d/ms-step=$(printf %06d ${5:-$4}).ckpt
}
link A muon_sweep/x-d9-c1-lr3p40e-3 d640 133600
link C muon_sweep/x-d10-c4-lr2p72e-2 d768 21248
link B muon_sweep/x-d9-c4-lr6p80e-3 d640 33408
link D muon_fp8recipe/fp8-d20-bf16 d1536 13600
link same muon_sweep/x-d9-c1-lr3p40e-3 d640 133600 1000
link same muon_sweep/x-d9-c4-lr6p80e-3 d640 33408 2000
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -lms 2000 > $S/gpu_use.csv 2>/dev/null &
SMI=$!
W5=$WC/project/gpic-wt/eval_official_run6/jobs
t0=$(date +%s)
bash $W5/evals_group.sh --wt eval_official_run6 smoke/$TAGN/A:d640:640:9:2:133600 smoke/$TAGN/C:d768:768:10:2:21248 smoke/$TAGN/D:d1536:1536:16:4:13600 smoke/$TAGN/B:d640:640:9:2:33408
echo "group rc=$?"
t1=$(date +%s)
bash $W5/evals_gmflow.sh 0 d640 640 9 2 $S/same --wt eval_official_run6 1000:euler:50 2000:euler:50
echo "evals_gmflow pipeline rc=$?"
t2=$(date +%s)
kill $SMI 2>/dev/null
source $WC/envs/pixgen/bin/activate
python - $S $RV/smoke/$TAGN $RV/smoke/run5_1109_0004 $WC/runs/gpic-speed/eval_smoke/run4_1108_2332 $((t1 - t0)) $((t2 - t1)) <<'PY'
import csv, json, os, sys
S, G, G5, R4 = sys.argv[1:5]; wg, ws = map(int, sys.argv[5:7]); S5 = S.replace("/raw_view", "").rsplit("/", 1)[0] + "/run5_1109_0004"
rd = lambda d, f: json.load(open(f"{d}/{f}")) if os.path.exists(f"{d}/{f}") else {}
ev = lambda st: f"eval_{st:06d}_euler_calls50_official"
ref = {"A": (f"{G5}/A/exp_ladder_d640/{ev(133600)}", 133600, "d640"),
       "C": (f"{G5}/C/exp_ladder_d768/{ev(21248)}", 21248, "d768"),
       "D": (f"{R4}/fp8-d20-bf16/exp_ladder_d1536/{ev(13600)}", 13600, "d1536"),
       "B": (f"{G5}/B/exp_ladder_d640/{ev(33408)}", 33408, "d640")}
def show(label, new_d, old_d):
    a, b = rd(new_d, "metrics.json"), rd(old_d, "metrics.json")
    ai, bi = rd(new_d, "metrics_inception.json"), rd(old_d, "metrics_inception.json")
    p, q = rd(new_d, "phase_times.json"), rd(old_d, "phase_times.json")
    serial = q.get("predict_s", 0) + max(q.get("eval_s", 0), q.get("inception_s", 0))
    print(f"{label}: FD {a.get('fd', float('nan')):.3f} (ref {b.get('fd', float('nan')):.3f}), FID {ai.get('inception_fd', float('nan')):.3f} "
          f"(ref {bi.get('inception_fd', float('nan')):.3f}); exact {all(a.get(k) == b.get(k) for k in b if isinstance(b[k], float) and not k.endswith('_s'))}/"
          f"{all(ai.get(k) == bi.get(k) for k in bi if isinstance(bi[k], float) and not k.endswith('_s'))}; phases {p} (ref {q})")
    return serial
tot = 0
for k in ("A", "C", "D", "B"):
    old, st, size = ref[k]
    tot += show(f"group {k}", f"{G}/{k}/exp_ladder_{size}/{ev(st)}", old)
print(f"group wall {wg} s (run5 smoke: 630 s for A, C, B without D)")
tot = 0
for st, k in ((1000, "A"), (2000, "B")):
    tot += show(f"same-root step {st} ({k})", f"{S}/same/exp_ladder_d640/{ev(st)}", f"{S5}/same/exp_ladder_d640/{ev(st)}")
print(f"same-root wall {ws} s (run5 smoke 417 s)")
rows = [r for r in csv.reader(open(f"{S}/gpu_use.csv")) if len(r) == 4]
if rows:
    util = [float(r[2].strip().split()[0]) for r in rows]; mem = [float(r[3].strip().split()[0]) for r in rows]
    n = len(util)
    print(f"GPU samples {n}: share at 0% {sum(u == 0 for u in util) / n:.3f}, mean util {sum(util) / n:.1f}%, peak memory {max(mem):.0f} MiB")
PY
rm -f $RV/smoke/$TAGN/*/exp_ladder_*/ms-step=*.ckpt   # the smoke's own raw views (0.8-1.3 GB each); the scores stay
echo EVAL_SMOKE_RUN6_DONE $S
