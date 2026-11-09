#!/bin/bash
#SBATCH -J gpic-gmflow-eval
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:30:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# GPIC evaluation of one GMFlow checkpoint (docs/gmflow.md) with one sampler at one number of network calls: 50K test captions, the official noise (EVAL_NOISE below), EMA net, no CFG, FD-DINOv2 / PRDC / MMD
# (the protocol of jobs/eval_energy.sh). K = 0 evaluates the point model w1147 (runs/gpic-speed/point_nodrop_d640) with the same samplers.
# usage: eval_gmflow.sh <K: 0 or the run's K> <step> <sampler: gm_sde | gm_sde_final_mean | gm_ode | gm_sde2 | gm_ode2 | gm_ode_cal | euler_mean | euler> <calls>
#   gm_sde: GM-SDE, first order, the last step returns a draw (for K = 0: DDPM with the small variance). gm_sde_final_mean: the last step returns the mixture mean.
#   gm_ode: GM-ODE, first order, ceil(128 / calls) analytic sub-steps per call (3 at 50, 16 at 8). euler_mean / euler: the official Euler on the mixture mean (for K = 0: the official Euler).
#   gm_sde2 / gm_ode2: GM-SDE 2 / GM-ODE 2 (order 2, docs/gmflow.md section 7); gm_ode2 keeps ceil(128 / calls) sub-steps, at least 2. K > 0 only.
# Output: <run>/eval_<step>_<sampler>_calls<calls>[_n<substeps>][_official]/metrics.json (eval_dir below). It never overwrites a finished result; a directory without metrics.json is moved aside.
# Env: LADDER_RUN_ROOT (run root); EVAL_SIZE, EVAL_D, EVAL_NB, EVAL_NT (run directory exp_ladder_<size>, width and blocks; default d640, 640, 9, 2; jobs/evals_gmflow.sh sets them);
# EVAL_SPREAD (image | pixel, default image: gm_spread of the model; the checkpoint must hold the per-pixel offset head if and only if it is pixel);
# EVAL_DEVICES (default 8): the GPUs one evaluation uses, for prediction and for the FD (jobs/evals_gmflow.sh --par runs one evaluation per GPU).
# EVAL_LIMIT_BATCHES (smoke test only: predict only that many batches per rank, keep the samples, skip the FD).
# EVAL_NOISE (official | prompt, default official; owner 2026-11-06: the official evaluation setup, except that CFG stays 1): official is the official noise, one
# noise tensor for all prompts (gpic_jsonl_dataset noise_per_prompt false), and adds _official to the output directory; prompt is each prompt's own noise, the
# protocol of every result before 2026-11-06, with the old directory names.
# docs/gmflow_hn.md: gm_ode_cal (K = 0 only): GM-ODE on the point model with the spread table EVAL_SPREAD_TABLE (required; tools/gmflow/calibrate_spread.py), output
# eval_<step>_gm_ode_cal-<first 8 hex of the table file's sha1>_calls<calls>. The table's ckpt must be ms-step=<step>, and at every call time t > 0 its s_x(t) must be
# at most (1 - t) / t (owner 2026-11-06: no clipped or scaled tables; gm_spread.py check). EVAL_SUBSTEPS (gm_ode, gm_ode2 and gm_ode_cal only) overrides the
# sub-steps per call, and the output directory then gets _n<substeps>. A GM-ODE with calls x sub-steps > 512 is refused: gm_convert's clamp norm >= 1e-6 shrinks the
# means at the last sub-steps beyond about 1000 (review_gmflow_hn F5); the clamp stays, so finished evaluations keep their numbers.
# Code: GMFLOW_WT (required: the frozen worktree; jobs/evals_gmflow.sh --wt sets it). A wq job cannot take EVAL_* or GMFLOW_WT from the submit shell, so submit
# evaluations through jobs/evals_gmflow.sh (--wt, --spread pixel, --table), not this script directly.
set -eo pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
K=$1 STEP=$(printf %06d $2) SAMPLER=$3 CALLS=$4
[ $# -eq 4 ] || { echo "usage: eval_gmflow.sh <K> <step> <sampler> <calls>"; exit 2; }
[ -n "${GMFLOW_WT:-}" ] || { echo "GMFLOW_WT is not set: name the frozen worktree (jobs/evals_gmflow.sh --wt sets it); the development worktree moves"; exit 2; }
WT=$GMFLOW_WT   # evals_gmflow.sh, train_gmflow.sh, run_gmflow.sh and eval_point_sampler.sh name the frozen worktree
eval_dir() {   # <step, 6 digits> <sampler> <calls> <substeps or ""> <table or ""> <noise tag: official or "">: the output directory's name. The same text is in eval_gmflow.sh and evals_gmflow.sh (T27).
  local s=$2; [ "$2" != gm_ode_cal ] || s=gm_ode_cal-$(sha1sum < "$5" | cut -c1-8)
  echo "eval_$1_${s}_calls$3${4:+_n$4}${6:+_$6}"
}
case $K in
  0) DEF_ROOT=$WC/runs/gpic-speed/point_nodrop_d640 ;;
  [1-9]|[1-9][0-9]) DEF_ROOT=$WC/runs/gpic-speed/gmflow_k${K}_d640 ;;
  *) echo "K must be 0 to 99"; exit 2 ;;
esac
SIZE=${EVAL_SIZE:-d640} WD=${EVAL_D:-640} NB=${EVAL_NB:-9} NT=${EVAL_NT:-2} SPREAD=${EVAL_SPREAD:-image}
case $SPREAD in image|pixel) ;; *) echo "EVAL_SPREAD must be image or pixel"; exit 2 ;; esac
NOISE=${EVAL_NOISE:-official}
case $NOISE in official) NPP=false NTAG=official ;; prompt) NPP=true NTAG="" ;; *) echo "EVAL_NOISE must be official or prompt"; exit 2 ;; esac
if [ $K -eq 0 ] && [ $SPREAD = pixel ]; then echo "the per-pixel spread needs a mixture (K > 0)"; exit 2; fi
SUBSTEPS=$(( (128 + CALLS - 1) / CALLS )); [ $SUBSTEPS -ge 2 ] || SUBSTEPS=2
NSUF=""
if [ -n "${EVAL_SUBSTEPS:-}" ]; then
  case $SAMPLER in gm_ode|gm_ode2|gm_ode_cal) ;; *) echo "EVAL_SUBSTEPS is for gm_ode, gm_ode2 and gm_ode_cal"; exit 2 ;; esac
  [[ $EVAL_SUBSTEPS =~ ^[1-9][0-9]*$ ]] || { echo "EVAL_SUBSTEPS must be a positive integer"; exit 2; }
  SUBSTEPS=$EVAL_SUBSTEPS NSUF=$EVAL_SUBSTEPS
fi
case $SAMPLER in gm_ode|gm_ode2|gm_ode_cal)
  [ $((CALLS * SUBSTEPS)) -le 512 ] || { echo "calls x sub-steps = $((CALLS * SUBSTEPS)) > 512: gm_convert's clamp shrinks the means there (review_gmflow_hn F5)"; exit 2; } ;;
esac
TABLE=""
case $SAMPLER in
  gm_sde) SARGS=(--config configs_h100/gmflow_sde_sample.yaml) ;;
  gm_sde_final_mean) SARGS=(--config configs_h100/gmflow_sde_sample.yaml --model.diffusion_sampler.init_args.final_mean true) ;;
  gm_ode) SARGS=(--config configs_h100/gmflow_ode_sample.yaml --model.diffusion_sampler.init_args.substeps $SUBSTEPS) ;;
  gm_sde2) SARGS=(--config configs_h100/gmflow_sde_sample.yaml --model.diffusion_sampler.init_args.order 2) ;;
  gm_ode2) SARGS=(--config configs_h100/gmflow_ode_sample.yaml --model.diffusion_sampler.init_args.substeps $SUBSTEPS --model.diffusion_sampler.init_args.order 2) ;;
  euler_mean|euler) SARGS=(--config configs_h100/gmflow_euler_sample.yaml) ;;
  gm_ode_cal)
    [ $K -eq 0 ] || { echo "gm_ode_cal is for a point model (K = 0); use gm_ode"; exit 2; }
    [ -f "${EVAL_SPREAD_TABLE:-}" ] || { echo "gm_ode_cal needs EVAL_SPREAD_TABLE=<table.json>"; exit 2; }
    TABLE=$(realpath "$EVAL_SPREAD_TABLE")   # the prediction runs after a cd (review_gmflow_hn F8)
    [ -f "$WT/baselines/PixelGen/src/diffusion/flow_matching/gm_spread.py" ] || { echo "$WT has no spread-table code"; exit 2; }
    SARGS=(--config configs_h100/gmflow_ode_sample.yaml --model.diffusion_sampler.init_args.substeps $SUBSTEPS
           --model.diffusion_sampler.init_args.spread_table "$TABLE") ;;
  *) echo "unknown sampler $SAMPLER"; exit 2 ;;
esac
case $K:$SAMPLER in 0:gm_ode|0:gm_sde_final_mean|0:gm_sde2|0:gm_ode2) echo "$SAMPLER needs a mixture (K > 0)"; exit 2 ;; esac   # order 2 on a point model is order 1
if [ $K -gt 0 ] && [ $SAMPLER = euler ]; then echo "use euler_mean for K > 0"; exit 2; fi
[ -f "$WT/baselines/PixelGen/src/diffusion/flow_matching/sampling_gmflow.py" ] || { echo "$WT holds old code"; exit 2; }
if [ $SPREAD = pixel ] || [ ${SAMPLER%2} != $SAMPLER ]; then
  grep -q "def gm_second_order" $WT/baselines/PixelGen/src/diffusion/flow_matching/gmflow.py || { echo "$WT has no per-pixel spread or second-order sampler"; exit 2; }
fi
RUN=${LADDER_RUN_ROOT:-$DEF_ROOT}/exp_ladder_$SIZE
CKPT=$(ls -t $RUN/ms-*$STEP*.ckpt 2>/dev/null | head -1); [ -n "$CKPT" ] || { echo "no ms checkpoint for step $STEP in $RUN"; exit 2; }
OUT=$RUN/$(eval_dir $STEP $SAMPLER $CALLS "$NSUF" "$TABLE" "$NTAG")
# gpic-28 2026-11-09: EVAL_PHASE all (default), or predict / score: jobs/evals_gmflow.sh pipelines a job's tasks, task i's scoring beside
# task i+1's predict (whose start-up and first compile leave the GPUs idle); predict leaves the samples in EVAL_SHM_ROOT for score
PHASE=${EVAL_PHASE:-all}
case $PHASE in all|predict|score) ;; *) echo "EVAL_PHASE must be all, predict or score"; exit 2 ;; esac
[ ! -f $OUT/metrics.json ] || { echo "$OUT/metrics.json exists; not overwriting"; exit 3; }
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD
if [ $PHASE != score ]; then
[ ! -d $OUT ] || mv $OUT $OUT.unfinished_$(date +%m%d_%H%M%S)
if [ $SAMPLER = gm_ode_cal ]; then   # the table belongs to this checkpoint and stays under the bound at these calls (F1, F2)
  python -m src.diffusion.flow_matching.gm_spread check "$TABLE" $STEP $CALLS || exit 2
fi
# the checkpoint loads with strict=False, so a wrong head or size would give a wrong FD and no error
python - "$CKPT" $K $WD $NB $NT $SPREAD <<'PY'
import re, sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)["state_dict"]
k, d, nb, nt = map(int, sys.argv[2:6])
spread = sys.argv[6]
out = sd["ema_denoiser.final_layer.linear.weight"].shape[0]
n = lambda part: len({m.group(1) for key in sd for m in [re.match(rf"ema_denoiser\.{part}\.(\d+)\.", key)] if m})
got = (sd["ema_denoiser.y_pos_embedding"].shape[-1], n("blocks"), n("text_refine_blocks"))
assert got == (d, nb, nt), f"checkpoint has d, nb, nt = {got}, arguments {(d, nb, nt)}"
assert out == (k * 4 * 256 if k else 768), f"final layer has {out} outputs, K = {k}"
assert any(n.startswith("ema_denoiser.gm_logstd") for n in sd) == (k > 0), "s-MLP presence does not match K"
assert ("ema_denoiser.gm_spread_head.weight" in sd) == (spread == "pixel"), f"per-pixel offset head presence does not match EVAL_SPREAD={spread}"
PY
mkdir -p $OUT
echo "ckpt=$CKPT code=$(git -C $WT rev-parse HEAD) K=$K spread=$SPREAD sampler=$SAMPLER calls=$CALLS substeps=$SUBSTEPS noise=$NOISE" | tee $OUT/ckpt.txt
[ $SAMPLER != gm_ode_cal ] || { echo "spread_table=$TABLE sha1=$(sha1sum < "$TABLE" | cut -c1-40)" | tee -a $OUT/ckpt.txt; cp "$TABLE" $OUT/spread_table.json; }
SPREAD_ARGS=(); [ $SPREAD = image ] || SPREAD_ARGS=(--model.denoiser.init_args.gm_spread $SPREAD)   # image: the command line of before this option
LIMIT=(); [ -z "${EVAL_LIMIT_BATCHES:-}" ] || LIMIT=(--trainer.limit_predict_batches $EVAL_LIMIT_BATCHES)
# gpic-28 2026-11-08 (w2353, the fixed cost of a predict): the text encoder runs eager (qwen3_text_encoder.py: compiled, it recompiled
# between the batch-1 null prompt and the batch-32 captions), and the 50K samples (9.8 GB) go to node memory instead of GPFS (writing
# them took ~25 s, and the check, DINOv2 and Inception passes read them again). The memory directory is removed at exit (the EXIT
# trap keeps env.sh's); leftovers of a killed evaluation older than 2 h are swept first.
export GPIC_EVAL_TE_EAGER=1
PRED_ROOT=$OUT/pred
if [ $PHASE = predict ]; then   # the samples must outlive this script: node memory only in the caller's EVAL_SHM_ROOT (removed by the
  if [ -d "${EVAL_SHM_ROOT:-}" ] && [ "$(df --output=avail -BG /dev/shm | tail -1 | tr -dc 0-9)" -gt 40 ]; then   # score phase, and at the caller's exit)
    SHM_DIR=$(mktemp -d $EVAL_SHM_ROOT/task_XXXXXX); PRED_ROOT=$SHM_DIR/pred
  fi
elif [ -d /dev/shm ] && [ "$(df --output=avail -BG /dev/shm | tail -1 | tr -dc 0-9)" -gt 40 ]; then
  find /dev/shm -maxdepth 1 -name 'gpic_eval_*' -user "$USER" -mmin +120 -exec rm -rf {} + 2>/dev/null || true
  SHM_DIR=$(mktemp -d /dev/shm/gpic_eval_XXXXXX)
  _prev_exit_cmd=$(trap -p EXIT | sed -E "s/^trap -- '(.*)' EXIT$/\1/; s/'\\\\''/'/g")
  _on_exit() { rm -rf -- "$SHM_DIR"; eval "$_prev_exit_cmd"; }
  trap _on_exit EXIT
  PRED_ROOT=$SHM_DIR/pred
fi
t0=$(date +%s)
WANDB_MODE=disabled python main.py predict --config configs_h100/lite_ladder_sample.yaml "${SARGS[@]}" --trainer.devices ${EVAL_DEVICES:-8} \
  --trainer.default_root_dir "$PRED_ROOT" --ckpt_path "$CKPT" --model.diffusion_sampler.init_args.num_steps $CALLS \
  --model.denoiser.init_args.hidden_size $WD --model.denoiser.init_args.num_groups $((WD / 64)) \
  --model.denoiser.init_args.num_blocks $NB --model.denoiser.init_args.num_text_blocks $NT --model.denoiser.init_args.gm_k $K "${SPREAD_ARGS[@]}" \
  --data.pred_dataset.init_args.noise_per_prompt $NPP "${LIMIT[@]}" > $OUT/predict.log 2>&1
t1=$(date +%s)
NPZ=$(find "$PRED_ROOT" -name output.npz | head -1)
[ -n "$NPZ" ] || { echo "no output.npz"; exit 4; }
if [ "$PRED_ROOT" != "$OUT/pred" ]; then   # everything but the samples back beside the metrics, as before
  mkdir -p "$OUT/pred" && (cd "$PRED_ROOT" && find . -type f ! -name output.npz -exec cp --parents {} "$OUT/pred/" \;)
fi
if [ -n "${EVAL_LIMIT_BATCHES:-}" ]; then
  python - "$NPZ" <<'PY' | tee $OUT/sample_check.txt
import sys, numpy as np
a = np.load(sys.argv[1])["arr_0"]
f = a.reshape(len(a), -1).astype(np.float32)
print(f"samples {a.shape} {a.dtype}; mean {a.mean():.2f}; std over images of the mean {f.mean(1).std():.2f}; mean within-image std {f.std(1).mean():.2f}; share of saturated pixels {np.mean((a == 0) | (a == 255)):.4f}")
assert a.ndim == 4 and len(a) >= 64 and a.std() > 5, "samples look broken"
PY
  echo "{\"predict_s\": $((t1-t0))}" | tee $OUT/phase_times.json
  exit 0
fi
if [ $PHASE = predict ]; then   # the score phase takes the samples from here
  echo "$NPZ" > $OUT/.npz_path; echo "${SHM_DIR:-}" > $OUT/.shm_dir; echo "$t0 $t1" > $OUT/.predict_t
  exit 0
fi
ts=$t1
else
  NPZ=$(cat $OUT/.npz_path 2>/dev/null || true)
  [ -n "$NPZ" ] && [ -f "$NPZ" ] || { echo "no samples for $OUT: run EVAL_PHASE=predict first"; exit 4; }
  read -r t0 t1 < $OUT/.predict_t
  SHM_DIR=$(cat $OUT/.shm_dir 2>/dev/null || true)
  ts=$(date +%s)   # the scoring's own start: it may wait for the task before it
fi
# the official evaluation's Inception metrics (FD, PRDC, MMD) on the same samples, before they are deleted (owner 2026-11-07: record every
# metric the official evaluation gives); a failure here leaves the DINOv2 result in place and says so. It runs beside the DINOv2
# evaluation, on the last visible GPU with TF's memory growth on, so no GPU waits on the one-GPU Inception pass (genmodel 2026-11-08
# night; owner: 「一定要处理空闲」, 「空闲gpu会有非常严重的后果」); the samples and both metric files are the same as before.
IG=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}; IG=${IG##*,}
( XDG_CACHE_HOME=$WC/cache/gpic_eval_xdg TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=$IG $WC/envs/gpic-eval-tf/bin/python $WT/tools/official_inception.py \
    "$NPZ" $WC/data/gpic/reference_stats/test_stats.npz $OUT/metrics_inception.json > $OUT/inception.log 2>&1 \
    || echo "inception metrics failed (see $OUT/inception.log)"
  date +%s > $OUT/.inception_end ) &
IPID=$!
if ! python -m torch.distributed.run --standalone --nproc_per_node ${EVAL_DEVICES:-8} $WT/tools/fast_eval.py "$NPZ" \
  $WC/data/gpic/reference_stats/test_stats.npz $OUT/metrics.json --workers ${EVAL_FAST_WORKERS:-6} > $OUT/eval.log 2>&1; then   # 6 x 8 ranks (was 11): the Inception pass beside it starved for CPU (58 -> 100 s)
  kill $IPID 2>/dev/null || true; wait $IPID 2>/dev/null || true; exit 1
fi
t2=$(date +%s)
echo "{\"predict_s\": $((t1-t0)), \"eval_s\": $((t2-ts))}" | tee $OUT/phase_times.json
cat $OUT/metrics.json
wait $IPID || true
t3=$(cat $OUT/.inception_end 2>/dev/null || date +%s); rm -f $OUT/.inception_end
PIPE=""; [ $PHASE != score ] || PIPE=", \"score_wait_s\": $((ts-t1)), \"pipelined\": 1"   # score_wait_s: predict end to scoring start
echo "{\"predict_s\": $((t1-t0)), \"eval_s\": $((t2-ts)), \"inception_s\": $((t3-ts)), \"inception_beside_eval\": 1$PIPE}" > $OUT/phase_times.json
# held-out loss (EMA and raw) of the evaluated checkpoint, once per checkpoint (tools/heldout_loss.py skips what is done)
if [ "${EVAL_HELDOUT:-0}" = 1 ]; then   # off by default: one checkpoint keeps 1 of 8 GPUs busy; batched held-out jobs (heldout_backfill.sh) run 8 checkpoints side by side
  ( HWT=$WC/project/gpic-wt/heldout_loss_run; export PYTHONPATH=$HWT/baselines/PixelGen WT=$HWT OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    source $HWT/jobs/heldout_loss_lib.sh
    HELDOUT_TRAIN_SHARDS="00787" heldout_run $WC/tmp/heldout_loss/heldout_loss.jsonl 3072 3072 ${EVAL_DEVICES:-8} "$CKPT" ) > $OUT/heldout.log 2>&1 \
    || echo "held-out loss failed (see $OUT/heldout.log)"
fi
rm -f "$NPZ"   # 50K x 196 KB; metrics.json and metrics_inception.json keep the result
if [ $PHASE = score ]; then [ -z "$SHM_DIR" ] || rm -rf -- "$SHM_DIR"; rm -f $OUT/.npz_path $OUT/.shm_dir $OUT/.predict_t; fi
