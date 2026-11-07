#!/bin/bash
#SBATCH -J gpic-full-gmflow
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=08:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# One budget segment of GMFlow on the full ladder (docs/gmflow.md section 17, owner 2026-11-06 21:4x): the full ladder's run of jobs/train_full.sh (the same
# recipe, data, shard seed 20261106, segments and LR schedule as the point model of that width) with the GMFlow head and loss of configs_h100/gmflow_train.yaml
# (per pixel, K components, lambda 0.5, one s_u per image), trained to <multiple> x <C0 steps>. <C0 steps> is GMFlow's FLOP-matched C0: the largest multiple of
# 800 with steps x (1 + r) <= the point model's C0 steps (docs/gmflow.md 17.2). All segments of a size share runs/gpic-speed/full_ladder_gmflow_k<K>/exp_ladder_<size>
# and the W&B run ladder-full-gmflow-k<K>-<size>; segment k resumes where segment k-1 stopped. ms checkpoints at the multiples of steps/4, last.ckpt every C0/8.
# Only d640 to d1024: from d1280 train_ladder.sh sends a run to the point model's MFU worktree, which has no GMFlow, so wider sizes are refused here.
# Code: the frozen worktree gpic-wt/ladder_full_gmflow_run (branch ladder-full-gmflow); the point model's gpic-wt/ladder_full_run is not touched.
# Evaluations: gpic-wt/gmflow_run7/jobs/evals_gmflow.sh <K> ... with LADDER_RUN_ROOT this root.
# usage: train_full_gmflow.sh <K> <size> <d> <nb> <nt> <C0 steps> <multiple>
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
[ $# -eq 7 ] || { echo "usage: train_full_gmflow.sh <K> <size> <d> <nb> <nt> <C0 steps> <multiple>"; exit 2; }
K=$1 SIZE=$2 WD=$3 NB=$4 NT=$5 S=$6 M=$7
[[ $K =~ ^[1-9][0-9]?$ ]] || { echo "K must be 1 to 99"; exit 2; }
[[ $WD =~ ^[0-9]+$ ]] && [ "$WD" -lt 1280 ] || { echo "GMFlow on the full ladder runs d640 to d1024: from d1280 train_ladder.sh would run the MFU worktree, which has no GMFlow"; exit 2; }
[ "$SIZE" = "d$WD" ] || { echo "size $SIZE does not match d $WD"; exit 2; }
case $M in 1|2|4|8|16) ;; *) echo "multiple must be 1, 2, 4, 8 or 16"; exit 2 ;; esac
[[ $S =~ ^[0-9]+$ ]] && [ $((S % 800)) -eq 0 ] && [ "$S" -gt 0 ] || { echo "C0 steps must be a positive multiple of 800"; exit 2; }
WT=${FULL_WT:-$WC/project/gpic-wt/ladder_full_gmflow_run}   # FULL_WT: smoke tests only
[ -e "$WT/.git" ] || { echo "no worktree $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
grep -q "gm_k" $WT/baselines/PixelGen/src/models/transformer/JiT_T2I.py && [ -f $WT/baselines/PixelGen/configs_h100/gmflow_train.yaml ] || { echo "$WT has no GMFlow"; exit 2; }
export LADDER_WT=$WT LADDER_RUN_ROOT=${FULL_ROOT:-$WC/runs/gpic-speed/full_ladder_gmflow_k$K} LADDER_LAST_EVERY=$((S / 8))
# The point model's shard seed (jobs/train_full.sh), so GMFlow reads the same permutation of the same shards as the point model of every width.
FULL_SHARD_SEED=20261106
mkdir -p $LADDER_RUN_ROOT/exp_ladder_$SIZE
[ -s $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt ] || echo $FULL_SHARD_SEED > $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt
# The GMFlow overlay comes first: it sets only the head (gm_k) and the trainer, and the full ladder's data and logger arguments after it keep their values.
exec bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $((S * M)) \
  --config configs_h100/gmflow_train.yaml --model.denoiser.init_args.gm_k $K \
  --data.train_dataset.init_args.tar_pattern 'gpic_train_{00000..07999}.tar' \
  --trainer.logger.init_args.name ladder_full_gmflow_k${K}_$SIZE --trainer.logger.init_args.id ladder-full-gmflow-k$K-$SIZE
