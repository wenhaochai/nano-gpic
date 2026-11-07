#!/bin/bash
#SBATCH -J gpic-repeat
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=08:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# One repetition arm (owner 2026-11-07): the Full ladder's recipe and first segment (jobs/train_full.sh ... 1) on a subset of
# GPIC train, the first <shards> shards (80: about 1.01M images, 8: about 101K), width d, trained to C0 steps. ms checkpoints at
# C0/4, C0/2, 3C0/4 and C0; last.ckpt every C0/8 steps. Root runs/gpic-speed/repeat_s<shards>/exp_ladder_<size>, W&B run
# ladder-repeat-s<shards>-<size>, the ladder's shard seed 20261106. With fewer shards than 8 GPUs x 32 data workers, every pass
# reads k copies of every shard, each from its own start sample, so all 32 workers of every rank read (EpochShuffledShards in
# src/data/dataset/gpic_dataset.py; 80 shards: k 16, 8 shards: k 32). Everything else is the Full ladder's: crop, caption grouping,
# text cache, LR schedule, EMA, hooks (jobs/train_ladder.sh, and its code by width: d1280 and wider would go to the MFU worktree,
# which has no shard copies, so this script stops them).
# Code: the frozen worktree gpic-wt/repeat_run (this branch's reviewed commit; REPEAT_WT for smoke tests). Evaluations:
# gpic-wt/gmflow_run4/jobs/evals_gmflow.sh (tools/submit_repeat.sh).
# usage: train_repeat.sh <size> <d> <nb> <nt> <C0 steps> <shards: 80 or 8> [main.py arguments: smoke tests only]
# Env (smoke tests only): REPEAT_WT (code worktree), REPEAT_ROOT (run root), REPEAT_LAST_EVERY (steps between last.ckpt saves).
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
[ $# -ge 6 ] || { echo "usage: train_repeat.sh <size> <d> <nb> <nt> <C0 steps> <shards: 80 or 8>"; exit 2; }
SIZE=$1 WD=$2 NB=$3 NT=$4 S=$5 N=$6; shift 6
case $N in 80|8) ;; *) echo "shards must be 80 or 8"; exit 2 ;; esac
[ $((S % 800)) -eq 0 ] || { echo "C0 steps must be a multiple of 800"; exit 2; }
[ "$WD" -lt 1280 ] 2>/dev/null || { echo "d$WD would run on the MFU worktree, which has no shard copies"; exit 2; }
WT=${REPEAT_WT:-$WC/project/gpic-wt/repeat_run}
[ -e "$WT/.git" ] || { echo "no worktree $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
grep -q "def _repeat_copies" $WT/baselines/PixelGen/src/data/dataset/gpic_dataset.py || { echo "$WT has no shard copies"; exit 2; }
export LADDER_WT=$WT LADDER_RUN_ROOT=${REPEAT_ROOT:-$WC/runs/gpic-speed/repeat_s$N} LADDER_LAST_EVERY=${REPEAT_LAST_EVERY:-$((S / 8))}
# The Full ladder's shard seed (jobs/train_full.sh), written before the run's first launch; train_ladder.sh keeps it from then on.
REPEAT_SHARD_SEED=20261106
mkdir -p $LADDER_RUN_ROOT/exp_ladder_$SIZE
[ -s $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt ] || echo $REPEAT_SHARD_SEED > $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt
exec bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $S \
  --data.train_dataset.init_args.tar_pattern "gpic_train_{00000..$(printf %05d $((N - 1)))}.tar" \
  --trainer.logger.init_args.name ladder-repeat-s$N-$SIZE --trainer.logger.init_args.id ladder-repeat-s$N-$SIZE "$@"
