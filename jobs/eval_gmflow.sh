#!/bin/bash
#SBATCH -J gpic-gmflow-eval
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:30:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# GPIC evaluation of one GMFlow checkpoint (docs/gmflow.md) with one sampler at one number of network calls: 50K test captions, each prompt's own noise, EMA net, no CFG, FD-DINOv2 / PRDC / MMD
# (the protocol of jobs/eval_energy.sh). K = 0 evaluates the point model w1147 (runs/gpic-speed/point_nodrop_d640) with the same samplers.
# usage: eval_gmflow.sh <K: 0 or the run's K> <step> <sampler: gm_sde | gm_sde_final_mean | gm_ode | gm_sde2 | gm_ode2 | euler_mean | euler> <calls>
#   gm_sde: GM-SDE, first order, the last step returns a draw (for K = 0: DDPM with the small variance). gm_sde_final_mean: the last step returns the mixture mean.
#   gm_ode: GM-ODE, first order, ceil(128 / calls) analytic sub-steps per call (3 at 50, 16 at 8). euler_mean / euler: the official Euler on the mixture mean (for K = 0: the official Euler).
#   gm_sde2 / gm_ode2: GM-SDE 2 / GM-ODE 2 (order 2, docs/gmflow.md section 7); gm_ode2 keeps ceil(128 / calls) sub-steps, at least 2. K > 0 only.
# Output: <run>/eval_<step>_<sampler>_calls<calls>/metrics.json. It never overwrites a finished result; a directory without metrics.json is moved aside.
# Env: LADDER_RUN_ROOT (run root); EVAL_SIZE, EVAL_D, EVAL_NB, EVAL_NT (run directory exp_ladder_<size>, width and blocks; default d640, 640, 9, 2; jobs/evals_gmflow.sh sets them);
# EVAL_SPREAD (image | pixel, default image: gm_spread of the model; the checkpoint must hold the per-pixel offset head if and only if it is pixel);
# EVAL_LIMIT_BATCHES (smoke test only: predict only that many batches per rank, keep the samples, skip the FD).
set -eo pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
K=$1 STEP=$(printf %06d $2) SAMPLER=$3 CALLS=$4
[ $# -eq 4 ] || { echo "usage: eval_gmflow.sh <K> <step> <sampler> <calls>"; exit 2; }
WT=${GMFLOW_WT:-$WC/project/gpic-wt/gmflow}   # run_gmflow.sh and eval_point_sampler.sh name the frozen worktree
case $K in
  0) DEF_ROOT=$WC/runs/gpic-speed/point_nodrop_d640 ;;
  [1-9]|[1-9][0-9]) DEF_ROOT=$WC/runs/gpic-speed/gmflow_k${K}_d640 ;;
  *) echo "K must be 0 to 99"; exit 2 ;;
esac
SIZE=${EVAL_SIZE:-d640} WD=${EVAL_D:-640} NB=${EVAL_NB:-9} NT=${EVAL_NT:-2} SPREAD=${EVAL_SPREAD:-image}
case $SPREAD in image|pixel) ;; *) echo "EVAL_SPREAD must be image or pixel"; exit 2 ;; esac
if [ $K -eq 0 ] && [ $SPREAD = pixel ]; then echo "the per-pixel spread needs a mixture (K > 0)"; exit 2; fi
SUBSTEPS=$(( (128 + CALLS - 1) / CALLS )); [ $SUBSTEPS -ge 2 ] || SUBSTEPS=2
case $SAMPLER in
  gm_sde) SARGS=(--config configs_h100/gmflow_sde_sample.yaml) ;;
  gm_sde_final_mean) SARGS=(--config configs_h100/gmflow_sde_sample.yaml --model.diffusion_sampler.init_args.final_mean true) ;;
  gm_ode) SARGS=(--config configs_h100/gmflow_ode_sample.yaml --model.diffusion_sampler.init_args.substeps $SUBSTEPS) ;;
  gm_sde2) SARGS=(--config configs_h100/gmflow_sde_sample.yaml --model.diffusion_sampler.init_args.order 2) ;;
  gm_ode2) SARGS=(--config configs_h100/gmflow_ode_sample.yaml --model.diffusion_sampler.init_args.substeps $SUBSTEPS --model.diffusion_sampler.init_args.order 2) ;;
  euler_mean|euler) SARGS=(--config configs_h100/gmflow_euler_sample.yaml) ;;
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
OUT=$RUN/eval_${STEP}_${SAMPLER}_calls${CALLS}
[ ! -f $OUT/metrics.json ] || { echo "$OUT/metrics.json exists; not overwriting"; exit 3; }
[ ! -d $OUT ] || mv $OUT $OUT.unfinished_$(date +%m%d_%H%M%S)
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD
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
echo "ckpt=$CKPT code=$(git -C $WT rev-parse HEAD) K=$K spread=$SPREAD sampler=$SAMPLER calls=$CALLS substeps=$SUBSTEPS" | tee $OUT/ckpt.txt
SPREAD_ARGS=(); [ $SPREAD = image ] || SPREAD_ARGS=(--model.denoiser.init_args.gm_spread $SPREAD)   # image: the command line of before this option
LIMIT=(); [ -z "${EVAL_LIMIT_BATCHES:-}" ] || LIMIT=(--trainer.limit_predict_batches $EVAL_LIMIT_BATCHES)
t0=$(date +%s)
WANDB_MODE=disabled python main.py predict --config configs_h100/lite_ladder_sample.yaml "${SARGS[@]}" --trainer.devices 8 \
  --trainer.default_root_dir "$OUT/pred" --ckpt_path "$CKPT" --model.diffusion_sampler.init_args.num_steps $CALLS \
  --model.denoiser.init_args.hidden_size $WD --model.denoiser.init_args.num_groups $((WD / 64)) \
  --model.denoiser.init_args.num_blocks $NB --model.denoiser.init_args.num_text_blocks $NT --model.denoiser.init_args.gm_k $K "${SPREAD_ARGS[@]}" \
  --data.pred_dataset.init_args.noise_per_prompt true "${LIMIT[@]}" > $OUT/predict.log 2>&1
t1=$(date +%s)
NPZ=$(find "$OUT/pred" -name output.npz | head -1)
[ -n "$NPZ" ] || { echo "no output.npz"; exit 4; }
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
python -m torch.distributed.run --standalone --nproc_per_node 8 $WT/tools/fast_eval.py "$NPZ" \
  $WC/data/gpic/reference_stats/test_stats.npz $OUT/metrics.json --workers 11 > $OUT/eval.log 2>&1
t2=$(date +%s)
echo "{\"predict_s\": $((t1-t0)), \"eval_s\": $((t2-t1))}" | tee $OUT/phase_times.json
cat $OUT/metrics.json
rm -f "$NPZ"   # 50K x 196 KB; metrics.json keeps the result
