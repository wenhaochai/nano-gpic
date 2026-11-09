#!/bin/bash
#SBATCH -J gpic-heldout-release
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --time=00:20:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# REL-HO (owner popup 2026-11-08 23:5x): the held-out loss of the released GPIC model (step 390600) on the ladder's metric, with the
# released model's own (official) text path; tools/heldout_release.py. Four processes side by side, one per weights (EMA, raw) and split
# (val, train shard 00787), then one line per weights appended to tmp/heldout_loss/heldout_loss.jsonl.
export NCCL_NVLS_ENABLE=0
set -eo pipefail
WC=/scratch/gpfs/GROUP/USER
WT=$WC/project/gpic-wt/heldout_release
source $WC/project/gpic/jobs/env.sh
export PYTHONPATH=$WT/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
J=${FETCHED_JOB_ID:-$SLURM_JOB_ID}; [ -n "$J" ] || { echo "no job id"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
echo "code $(git -C $WT rev-parse HEAD)"
P=$WC/tmp/heldout_loss/release_$J; mkdir -p $P
i=0; pids=()
for w in ema raw; do for s in val train; do
  python $WT/tools/heldout_release.py work --weights $w --split $s --gpu $i --out $P/$w-$s.json > $P/$w-$s.log 2>&1 &
  pids+=($!); i=$((i + 1))
done; done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
grep -h "^\[release\].*loss" $P/*.log || true
[ $fail = 0 ] || { echo "a part failed (see $P/*.log)"; exit 1; }
python $WT/tools/heldout_release.py merge $P/ema-val.json $P/ema-train.json $P/raw-val.json $P/raw-train.json
