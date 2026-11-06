#!/bin/bash
#SBATCH -J gpic-clip-share-point
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:30:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Review F4 (docs/review_gmflow_2026-11-05.md): how often the gradient clip at 1.0 acts on the point model. w1147 logged no gradient norm; the GMFlow smoke w1168
# was above the clip at 14 of 20 printed steps in its first 1000 steps. Two short runs of the w1147 recipe (lite_ladder.yaml, no caption dropout, the same shard
# seed and crop), with rank 0 printing the global pre-clip norm every 10 optimizer steps (GPIC_GRAD_PRINT):
#   A: from scratch to step 1000, the window of the smoke;
#   B: from a copy of w1147's last.ckpt (step 66816) to step 67216, the end of training.
# No W&B. Code: this worktree, gpic-wt/gmflow, with no local changes. It writes only under runs/gpic-speed/clip_share_point_<time>.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/gmflow
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
grep -q "GPIC_GRAD_PRINT" $WT/baselines/PixelGen/src/lightning_model_gpic.py || { echo "$WT holds old code"; exit 2; }
SRC=$WC/runs/gpic-speed/point_nodrop_d640/exp_ladder_d640/last.ckpt
[ -f $SRC ] || { echo "no $SRC"; exit 2; }
OUT=$WC/runs/gpic-speed/clip_share_point_$(date +%m%d_%H%M%S)
mkdir -p $OUT/a $OUT/b/exp_ladder_d640
git -C $WT rev-parse HEAD > $OUT/commit.txt
export LADDER_WT=$WT LADDER_PARTIAL_OK=1 WANDB_MODE=disabled GPIC_GRAD_PRINT=10
export GPIC_REGION_CROP=0 GPIC_SHARD_SEED=1730000000   # as w1147 (ar_real/jobs/run_point_nodrop.sh)
POINT=(--model.diffusion_trainer.init_args.null_condition_p 0.0)   # the gmflow branch's lite_ladder.yaml keeps 0.1; w1147's config has 0.0. Resolved config = w1147's but for the logger, max_steps, the root dir, gm_k 0 and head_width 0
log() { echo "[$(date +%T)] $*" | tee -a $OUT/phases.log; }
share() {   # <log>: the printed steps after the warm-up, the share above the clip, and the same on the smoke's steps (multiples of 50)
python3 -I - "$1" <<'PY' | tee -a $OUT/phases.log
import re, statistics, sys
s = open(sys.argv[1], errors="replace").read()
s = s.split("[warmup] done")[-1]
g = dict((int(a), float(b)) for a, b in re.findall(r"\[grad\] step (\d+): grad norm before the clip ([-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf))", s))
v = [g[k] for k in sorted(g)]
m = [g[k] for k in sorted(g) if k % 50 == 0 and k > 0]
print(f"{sys.argv[1]}: steps {min(g) if g else None}-{max(g) if g else None}, {len(v)} printed; above the clip 1.0: {sum(x > 1 for x in v)}/{len(v)}; "
      f"median {statistics.median(v) if v else None}; on multiples of 50: {sum(x > 1 for x in m)}/{len(m)}")
print("  by step:", {k: round(g[k], 3) for k in sorted(g) if k % 50 == 0})
PY
}
fail=0
log "START A: from scratch to step 1000"
LADDER_RUN_ROOT=$OUT/a timeout -k 60 720 bash $WT/jobs/train_ladder.sh d640 640 9 2 66816 "${POINT[@]}" --trainer.max_steps 1000 > $OUT/a.log 2>&1
rc=$?; log "END A rc=$rc"; [ $rc -eq 0 ] || fail=1
share $OUT/a.log || fail=1
log "START B: from a copy of w1147's last.ckpt (step 66816) to step 67216"
cp $SRC $OUT/b/exp_ladder_d640/last.ckpt   # a copy, not a link: a save of last.ckpt must not reach w1147's file
LADDER_RUN_ROOT=$OUT/b timeout -k 60 720 bash $WT/jobs/train_ladder.sh d640 640 9 2 66816 "${POINT[@]}" --trainer.max_steps 67216 > $OUT/b.log 2>&1
rc=$?; log "END B rc=$rc"; [ $rc -eq 0 ] || fail=1
grep -q "resuming from $OUT/b/exp_ladder_d640/last.ckpt" $OUT/b.log || { log "B did not resume from the copy"; fail=1; }
share $OUT/b.log || fail=1
rm -f $OUT/b/exp_ladder_d640/*.ckpt   # 1.75 GB copies; the printed norms are the result
log "DONE $OUT fail=$fail"
exit $fail
