#!/bin/bash
#SBATCH -J gpic-gmflow-evals
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# A list of evaluations of one run (docs/gmflow.md section 6): jobs/eval_gmflow.sh for each <step>:<sampler>:<calls>, at any width; 50K test captions, the
# official noise (--noise below), EMA, no CFG. Finished tasks are skipped, so a rerun does only the rest.
# K = 0 evaluates a point model. --link <run dir> links that run's ms checkpoints into <root>/exp_ladder_<size>, so nothing is written in its directory (the ladder's
# own evaluations, official noise and Euler at 50 steps, have no other call counts, so the ladder's checkpoints are evaluated again here).
# usage: evals_gmflow.sh <K> <size> <d> <nb> <nt> <root> [--link <run dir>] [--wt <worktree>] [--spread pixel] [--table <json>] [--noise official|prompt] <step:sampler:calls[:substeps]> ...
# --noise: official (the default since 2026-11-06, owner: the official evaluation setup except CFG = 1; one noise for all prompts, directories end in _official) or
# prompt (each prompt's own noise, the protocol of every result before; the old directory names); eval_gmflow.sh EVAL_NOISE.
# --spread pixel: the run has the per-pixel spread (docs/gmflow.md section 7; eval_gmflow.sh EVAL_SPREAD); default image.
# docs/gmflow_hn.md: a 4th field sets EVAL_SUBSTEPS for that task (output _n<substeps>); --table sets EVAL_SPREAD_TABLE (sampler gm_ode_cal, output name with the
# table's hash, eval_dir below).
# Code: the frozen worktree gpic-wt/gmflow_run2, or gpic-wt/<worktree> with --wt; every evaluation checks that it still holds the commit of the start.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
[ $# -ge 7 ] || { echo "usage: evals_gmflow.sh <K> <size> <d> <nb> <nt> <root> [--link <dir>] [--wt <worktree>] [--spread pixel] [--table <json>] [--noise official|prompt] <tasks...>"; exit 2; }
K=$1 SIZE=$2 WD=$3 NB=$4 NT=$5 ROOT=$6; shift 6
LINK=""
if [ "$1" = "--link" ]; then LINK=$2; shift 2; [ -d "$LINK" ] || { echo "no run directory $LINK"; exit 2; }; fi
WTN=gmflow_run2
if [ "$1" = "--wt" ]; then WTN=$2; shift 2; [[ $WTN =~ ^[A-Za-z0-9_-]+$ ]] || { echo "--wt takes a worktree name under gpic-wt/"; exit 2; }; fi
SPREAD=image
if [ "$1" = "--spread" ]; then SPREAD=$2; shift 2; case $SPREAD in image|pixel) ;; *) echo "--spread must be image or pixel"; exit 2 ;; esac; fi
TABLE=""
if [ "$1" = "--table" ]; then TABLE=$2; shift 2; [ -f "$TABLE" ] || { echo "no table $TABLE"; exit 2; }; TABLE=$(realpath "$TABLE"); export EVAL_SPREAD_TABLE=$TABLE; fi
NOISE=official
if [ "$1" = "--noise" ]; then NOISE=$2; shift 2; case $NOISE in official|prompt) ;; *) echo "--noise must be official or prompt"; exit 2 ;; esac; fi
NTAG=""; [ $NOISE = prompt ] || NTAG=official
[ $# -ge 1 ] || { echo "no tasks"; exit 2; }
WT=$WC/project/gpic-wt/$WTN
[ -f "$WT/baselines/PixelGen/src/diffusion/flow_matching/sampling_gmflow.py" ] || { echo "$WT holds old code"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
SHA=$(git -C $WT rev-parse HEAD)
eval_dir() {   # <step, 6 digits> <sampler> <calls> <substeps or ""> <table or ""> <noise tag: official or "">: the output directory's name. The same text is in eval_gmflow.sh and evals_gmflow.sh (T27).
  local s=$2; [ "$2" != gm_ode_cal ] || s=gm_ode_cal-$(sha1sum < "$5" | cut -c1-8)
  echo "eval_$1_${s}_calls$3${4:+_n$4}${6:+_$6}"
}
RUN=$ROOT/exp_ladder_$SIZE
mkdir -p $RUN
for t in "$@"; do
  case $t in *:*:*) ;; *) echo "task $t is not step:sampler:calls"; exit 2 ;; esac
  case $t in *:gm_ode_cal:*) [ -n "$TABLE" ] || { echo "task $t needs --table"; exit 2; } ;; esac
  S6=$(printf %06d ${t%%:*})
  if [ -n "$LINK" ] && [ ! -e $RUN/ms-step=$S6.ckpt ]; then
    [ -f $LINK/ms-step=$S6.ckpt ] || { echo "no $LINK/ms-step=$S6.ckpt"; exit 2; }
    ln -s $LINK/ms-step=$S6.ckpt $RUN/
  fi
done
[ -f $ROOT/commit.txt ] || echo $SHA > $ROOT/commit.txt
export GMFLOW_WT=$WT LADDER_RUN_ROOT=$ROOT EVAL_SIZE=$SIZE EVAL_D=$WD EVAL_NB=$NB EVAL_NT=$NT EVAL_SPREAD=$SPREAD EVAL_NOISE=$NOISE
fail=0
for t in "$@"; do
  STEP=${t%%:*} rest=${t#*:}; S=${rest%%:*} C=${rest#*:}
  N=""; case $C in *:*) N=${C#*:}; C=${C%%:*} ;; esac   # step:sampler:calls:substeps
  D=$(eval_dir $(printf %06d $STEP) $S $C "$N" "$TABLE" "$NTAG")
  if [ -f $RUN/$D/metrics.json ]; then echo "eval $t done before"; continue; fi
  [ "$(git -C $WT rev-parse HEAD)" = "$SHA" ] || { echo "$WT moved from $SHA"; exit 2; }
  LOG=eval_${STEP}_${S}_$C${N:+_n$N}${NTAG:+_$NTAG}; [ $S != gm_ode_cal ] || LOG=eval_${STEP}_${D#eval_*_}   # the old log names (+ _official); a table's hash for gm_ode_cal
  EVAL_SUBSTEPS=$N bash $WT/jobs/eval_gmflow.sh $K $STEP $S $C > $ROOT/$LOG.log 2>&1 || fail=1
  echo "eval $t fail=$fail"
done
exit $fail
