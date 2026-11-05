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
# usage: eval_gmflow.sh <K: 8 | 1 | 0> <step> <sampler: gm_sde | gm_sde_final_mean | gm_ode | euler_mean | euler> <calls: 50 | 8>
#   gm_sde: GM-SDE, first order, the last step returns a draw (for K = 0: DDPM with the small variance). gm_sde_final_mean: the last step returns the mixture mean.
#   gm_ode: GM-ODE, first order, ceil(128 / calls) analytic sub-steps per call (3 at 50, 16 at 8). euler_mean / euler: the official Euler on the mixture mean (for K = 0: the official Euler).
# Output: <run>/eval_<step>_<sampler>_calls<calls>/metrics.json. It never overwrites a finished result; a directory without metrics.json is moved aside.
# Env (smoke test only): LADDER_RUN_ROOT (run root), EVAL_LIMIT_BATCHES (predict only that many batches per rank, keep the samples, skip the FD).
set -eo pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
K=$1 STEP=$(printf %06d $2) SAMPLER=$3 CALLS=$4
[ $# -eq 4 ] || { echo "usage: eval_gmflow.sh <K> <step> <sampler> <calls>"; exit 2; }
WT=$WC/project/gpic-wt/gmflow
case $K in
  8|1) DEF_ROOT=$WC/runs/gpic-speed/gmflow_k${K}_d640 ;;
  0) DEF_ROOT=$WC/runs/gpic-speed/point_nodrop_d640 ;;
  *) echo "K must be 8, 1 or 0"; exit 2 ;;
esac
SUBSTEPS=$(( (128 + CALLS - 1) / CALLS )); [ $SUBSTEPS -ge 2 ] || SUBSTEPS=2
case $SAMPLER in
  gm_sde) SARGS=(--config configs_h100/gmflow_sde_sample.yaml) ;;
  gm_sde_final_mean) SARGS=(--config configs_h100/gmflow_sde_sample.yaml --model.diffusion_sampler.init_args.final_mean true) ;;
  gm_ode) SARGS=(--config configs_h100/gmflow_ode_sample.yaml --model.diffusion_sampler.init_args.substeps $SUBSTEPS) ;;
  euler_mean|euler) SARGS=(--config configs_h100/gmflow_euler_sample.yaml) ;;
  *) echo "unknown sampler $SAMPLER"; exit 2 ;;
esac
if [ $K -eq 0 ] && { [ $SAMPLER = gm_ode ] || [ $SAMPLER = gm_sde_final_mean ]; }; then echo "$SAMPLER needs a mixture (K > 0)"; exit 2; fi
if [ $K -gt 0 ] && [ $SAMPLER = euler ]; then echo "use euler_mean for K > 0"; exit 2; fi
[ -f "$WT/baselines/PixelGen/src/diffusion/flow_matching/sampling_gmflow.py" ] || { echo "$WT holds old code"; exit 2; }
RUN=${LADDER_RUN_ROOT:-$DEF_ROOT}/exp_ladder_d640
CKPT=$(ls -t $RUN/ms-*$STEP*.ckpt 2>/dev/null | head -1); [ -n "$CKPT" ] || { echo "no ms checkpoint for step $STEP in $RUN"; exit 2; }
OUT=$RUN/eval_${STEP}_${SAMPLER}_calls${CALLS}
[ ! -f $OUT/metrics.json ] || { echo "$OUT/metrics.json exists; not overwriting"; exit 3; }
[ ! -d $OUT ] || mv $OUT $OUT.unfinished_$(date +%m%d_%H%M%S)
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD
# the checkpoint loads with strict=False, so a wrong head would give a wrong FD and no error
python - "$CKPT" $K <<'PY'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)["state_dict"]
k = int(sys.argv[2])
out = sd["ema_denoiser.final_layer.linear.weight"].shape[0]
assert sd["ema_denoiser.y_pos_embedding"].shape[-1] == 640, "not a d640 checkpoint"
assert out == (k * 4 * 256 if k else 768), f"final layer has {out} outputs, K = {k}"
assert any(n.startswith("ema_denoiser.gm_logstd") for n in sd) == (k > 0), "s-MLP presence does not match K"
PY
mkdir -p $OUT
echo "ckpt=$CKPT code=$(git -C $WT rev-parse HEAD) K=$K sampler=$SAMPLER calls=$CALLS substeps=$SUBSTEPS" | tee $OUT/ckpt.txt
LIMIT=(); [ -z "${EVAL_LIMIT_BATCHES:-}" ] || LIMIT=(--trainer.limit_predict_batches $EVAL_LIMIT_BATCHES)
t0=$(date +%s)
WANDB_MODE=disabled python main.py predict --config configs_h100/lite_ladder_sample.yaml "${SARGS[@]}" --trainer.devices 8 \
  --trainer.default_root_dir "$OUT/pred" --ckpt_path "$CKPT" --model.diffusion_sampler.init_args.num_steps $CALLS \
  --model.denoiser.init_args.hidden_size 640 --model.denoiser.init_args.num_groups 10 \
  --model.denoiser.init_args.num_blocks 9 --model.denoiser.init_args.num_text_blocks 2 --model.denoiser.init_args.gm_k $K \
  --data.pred_dataset.init_args.noise_per_prompt true "${LIMIT[@]}" > $OUT/predict.log 2>&1
t1=$(date +%s)
NPZ=$(find "$OUT/pred" -name output.npz | head -1)
[ -n "$NPZ" ] || { echo "no output.npz"; exit 4; }
if [ -n "${EVAL_LIMIT_BATCHES:-}" ]; then
  python - "$NPZ" <<'PY' | tee $OUT/sample_check.txt
import sys, numpy as np
a = np.load(sys.argv[1])["arr_0"]
print(f"samples {a.shape} {a.dtype}; mean {a.mean():.2f}; std over images of the mean {a.reshape(len(a), -1).mean(1).std():.2f}; share of saturated pixels {np.mean((a == 0) | (a == 255)):.4f}")
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
