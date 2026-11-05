#!/bin/bash
#SBATCH -J gpic-ar-eta-real
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:45:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Test C: the jump process with eta 5 and 20 on the better real arm's weights at C0/8 (ar_few_d640, step 52320), no training. Test A's best values on 256 images. The rule stays on first64, set before the run; these numbers are reported beside it. Expected: about 25 min. Code: the frozen worktree gpic-wt/ar_real.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/ar_real
grep -q "def sample_eta" $WT/baselines/PixelGen/src/models/transformer/AR_JiT.py || { echo "$WT holds old code"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
export LADDER_WT=$WT LADDER_RUN_ROOT=$WC/runs/gpic-speed/ar_few_d640
fail=0
for V in eta5 eta20; do
  bash $WT/jobs/eval_ar.sh d640 640 9 2 52320 $V > $LADDER_RUN_ROOT/eval_052320_$V.log 2>&1 || fail=1
  echo "eval $V fail=$fail"
done
exit $fail
