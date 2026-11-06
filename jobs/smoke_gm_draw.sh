#!/bin/bash
#SBATCH -J gpic-gm-draw-smoke
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:40:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Smoke test of the draw sampler gm_draw (docs/gmflow.md section 14) through jobs/eval_gmflow.sh on 8 H100s: 2 batches per rank (512 prompts), no FD, on the real
# d640 checkpoints of K = 8 (31168) and K = 1 (33280), linked into a scratch root so nothing is written in their run directories; 8 and 50 calls. Fails on a
# nonzero exit or a missing sample check. Code: the frozen worktree named by the first argument (default gpic-wt/gmflow_run5), no local changes.
# usage: smoke_gm_draw.sh [worktree name under gpic-wt/]
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/${1:-gmflow_run5}
[ -e "$WT/.git" ] || { echo "no worktree $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
grep -q "class GMDrawSampler" $WT/baselines/PixelGen/src/diffusion/flow_matching/sampling_gmflow.py || { echo "$WT holds old code"; exit 2; }
OUT=$WC/runs/gpic-speed/gm_draw_smoke_$(date +%m%d_%H%M%S)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
export GMFLOW_WT=$WT WANDB_MODE=disabled EVAL_LIMIT_BATCHES=2
log() { echo "[$(date +%T)] $*" | tee -a $OUT/phases.log; }
fail=0
for spec in "8 gmflow_k8_d640 31168" "1 gmflow_k1_d640 33280"; do
  read -r K RUN STEP <<< "$spec"
  mkdir -p $OUT/k$K/exp_ladder_d640
  ln -s $WC/runs/gpic-speed/$RUN/exp_ladder_d640/ms-step=$(printf %06d $STEP).ckpt $OUT/k$K/exp_ladder_d640/ || fail=1
  for C in 8 50; do
    log "START K=$K step $STEP gm_draw $C calls"
    LADDER_RUN_ROOT=$OUT/k$K timeout -k 60 900 bash $WT/jobs/eval_gmflow.sh $K $STEP gm_draw $C > $OUT/eval_k${K}_c$C.log 2>&1
    rc=$?
    chk=$(cat $OUT/k$K/exp_ladder_d640/eval_$(printf %06d $STEP)_gm_draw_calls${C}*/sample_check.txt 2>/dev/null)
    log "END K=$K gm_draw $C calls rc=$rc sample_check: $chk"
    [ $rc -eq 0 ] && [ -n "$chk" ] || fail=1
  done
done
log "DONE $OUT fail=$fail"
exit $fail
