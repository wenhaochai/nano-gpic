#!/bin/bash
#SBATCH -J gpic-ladder
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=08:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# One run of the PixelGen ladder's iso-FLOP sweep (docs/ladder_plan.md, sizes and steps in docs/ladder_runs.json):
# configs_h100/lite_ladder.yaml (the official JiT-T2I recipe with grouped, masked text and a flat LR after warmup) at
# width d (heads d/64), nb main blocks and nt text blocks, max_steps steps of 256 images on 8 GPUs.
# Checkpoints: last.ckpt (full state) every max_steps/32 steps for resuming; ms-step=<step>.ckpt (model + EMA) at
# max_steps/4, /2, 3/4 and the end, for the FD evals (jobs/eval_ladder.sh).
# Code: the frozen worktree gpic-wt/ladder_run (the reviewed commit), so later edits cannot reach a running size.
# usage: train_ladder.sh <size> <d> <nb> <nt> <max_steps> [main.py arguments...]
# Resumable (wq --partial): resumes from the newest last*.ckpt; exits 0 only when max_steps are done.
# Env (smoke tests only): LADDER_WT (code worktree), LADDER_RUN_ROOT (run root), BENCH_OUT (adds the step timer),
# LADDER_PARTIAL_OK (a launch stopped early by --trainer.max_steps counts as done).
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
SIZE=$1 D=$2 NB=$3 NT=$4 STEPS=$5; shift 5
WT=${LADDER_WT:-$WC/project/gpic-wt/ladder_run}
ROOT=${LADDER_RUN_ROOT:-$WC/runs/gpic-speed}
RUN=$ROOT/exp_ladder_$SIZE   # main.py: <default_root_dir>/exp_<tags.exp>
if [ $((STEPS % 32)) -ne 0 ] || [ $((D % 64)) -ne 0 ]; then echo "max_steps must be a multiple of 32 and d of 64"; exit 2; fi
MS=$((STEPS / 4)); LAST=$((STEPS / 32))
mkdir -p $RUN
echo "$(date -u +%FT%TZ) $(git -C $WT rev-parse HEAD) size=$SIZE d=$D nb=$NB nt=$NT max_steps=$STEPS ms_every=$MS last_every=$LAST host=$(hostname) job=$JOBID" >> $RUN/launches.txt
# GPU health gate: a throttled GPU slows the whole run; exit 98 = requeue elsewhere
python $WT/tools/gpu_health.py > $RUN/gpu_health_$(date +%s).txt 2>&1 || { tail -3 $(ls -t $RUN/gpu_health_*.txt | head -1); echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
CKPT_ARGS=()
LASTCK=$(ls -t $RUN/last*.ckpt 2>/dev/null | head -1)
if [ -n "$LASTCK" ]; then CKPT_ARGS=(--ckpt_path "$LASTCK"); echo "resuming from $LASTCK"; fi
BENCH_CB=""
if [ -n "${BENCH_OUT:-}" ]; then BENCH_CB=',{"class_path":"src.callbacks.bench.BenchTimer"}'; fi
# the ms hook first: a kill between the two saves must not leave last.ckpt ahead of a missing ms file (review N1)
CB='[{"class_path":"src.callbacks.model_checkpoint.CheckpointHook","init_args":{"filename":"ms-{step:06d}","monitor":null,"save_last":false,"save_top_k":-1,"every_n_train_steps":'$MS'}},{"class_path":"src.callbacks.model_checkpoint.CheckpointHook","init_args":{"filename":"resume-{step:06d}","monitor":null,"save_last":true,"save_top_k":0,"every_n_train_steps":'$LAST'}},{"class_path":"src.callbacks.compile_warmup.CompileWarmup"}'$BENCH_CB']'
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD
export GPIC_SHARD_SEED=$(date +%s)   # shard order shuffled per launch and per epoch, shared by all ranks
python main.py fit --config configs_h100/lite_ladder.yaml --trainer.default_root_dir $ROOT \
  --trainer.max_steps $STEPS --trainer.callbacks "$CB" \
  --model.denoiser.init_args.hidden_size $D --model.denoiser.init_args.num_groups $((D / 64)) \
  --model.denoiser.init_args.num_blocks $NB --model.denoiser.init_args.num_text_blocks $NT \
  --tags.exp ladder_$SIZE --trainer.logger.init_args.name ladder_$SIZE --trainer.logger.init_args.id ladder-$SIZE \
  "$@" "${CKPT_ARGS[@]}"
rc=$?
# Lightning turns a SIGTERM that reaches only the ranks into exit 0: done means the final ms checkpoint exists (review S1)
if [ $rc -eq 0 ] && [ -z "${LADDER_PARTIAL_OK:-}" ] && ! ls $RUN/ms-step=$(printf %06d $STEPS)*.ckpt > /dev/null 2>&1; then
  echo "main.py exit 0 before max_steps $STEPS"; rc=99
fi
echo "main.py exit $rc"
exit $rc
