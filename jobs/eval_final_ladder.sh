#!/bin/bash
# The post's evaluation of one final-ladder checkpoint: the official GPIC setup except CFG = 1. 50,000 test captions, one sample each,
# the official noise (one noise tensor for all prompts), Euler at 50 steps, the conditional branch only; FD-DINOv2 with PRDC and MMD
# (tools/fast_eval.py: gpic_eval's DINOv2 model and metric code, sharded over the GPUs) and, with GPIC_TF_PYTHON set, the official
# Inception metrics (tools/official_inception_sharded.py, in a TensorFlow environment with gpic_eval). The post's main metric, the
# held-out loss, comes from tools/heldout_loss.py (README).
# A final-ladder checkpoint holds the raw weights in both of its weight slots (no EMA); the sampler reads the EMA slot, so the script
# checks that the two slots are equal before it samples.
# usage: GPIC_WC=<work root> bash jobs/eval_final_ladder.sh <d10|d11|d12|d13|d14|d16|d18> [step]   (one 8-GPU node; step: the last)
#   reads <work root>/runs/final_ladder/<size>/exp_ladder_d<width>/ms-step=<step>.ckpt (GPIC_OUT moves the runs; GPIC_CKPT names any
#   checkpoint of that size), <data>/gpic_test_50k.jsonl and <data>/reference_stats/test_stats.npz (tools/download_lite.py fetches
#   both); writes metrics.json (and metrics_inception.json) to eval_<step>_euler50_official/ beside the checkpoint (GPIC_EVAL_OUT moves it).
set -eo pipefail
SIZE=${1:?usage: eval_final_ladder.sh <size> [step]}; STEP=${2:-}
HERE=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
WC=${GPIC_WC:?set GPIC_WC to the work root (see the usage above)}
DATA=${GPIC_DATA:-$WC/data/gpic}
source $HERE/jobs/final_ladder_sizes.sh
final_ladder_size $SIZE || { echo "unknown size $SIZE: $FINAL_LADDER_SIZES"; exit 2; }
STEP=${STEP:-$S}
S6=$(printf %06d $((10#$STEP)))
CKPT=${GPIC_CKPT:-${GPIC_OUT:-$WC/runs/final_ladder}/$SIZE/exp_ladder_d$D/ms-step=$S6.ckpt}
[ -f "$CKPT" ] || { echo "no checkpoint $CKPT"; exit 2; }
for f in gpic_test_50k.jsonl reference_stats/test_stats.npz; do [ -f $DATA/$f ] || { echo "missing $DATA/$f (tools/download_lite.py)"; exit 2; }; done
export GPIC_WC=$WC GPIC_REPO=$HERE
source $HERE/jobs/env.sh
export NCCL_NVLS_ENABLE=0   # NCCL 2.26's NVLS all-gather can permute rank blocks (NVIDIA/nccl #1906)
OUT=${GPIC_EVAL_OUT:-$(dirname $CKPT)/eval_${S6}_euler50_official}
if [ -f $OUT/metrics.json ]; then echo "already done: $OUT/metrics.json"; cat $OUT/metrics.json; exit 0; fi
mkdir -p $OUT
echo "ckpt=$CKPT code=$(git -C $HERE rev-parse HEAD) size=$SIZE step=$STEP" | tee $OUT/ckpt.txt
cd $HERE/baselines/PixelGen
python - "$CKPT" <<'PY'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)["state_dict"]
raw = {k[len("denoiser."):]: v for k, v in sd.items() if k.startswith("denoiser.")}
ema = {k[len("ema_denoiser."):]: v for k, v in sd.items() if k.startswith("ema_denoiser.")}
bad = [k for k in raw if k not in ema or not torch.equal(raw[k], ema[k])]
assert raw and not bad, f"{len(bad)} of {len(raw)} tensors differ between the raw and the EMA slot: not a checkpoint of a run without the EMA"
print(f"the raw weights are in both slots ({len(raw)} tensors)")
PY
t0=$(date +%s)
WANDB_MODE=disabled python main.py predict --config configs_h100/lite_ladder_sample.yaml --trainer.devices 8 \
  --trainer.default_root_dir $OUT/pred --ckpt_path $CKPT --model.diffusion_sampler.init_args.num_steps 50 \
  --model.denoiser.init_args.hidden_size $D --model.denoiser.init_args.num_groups $((D / 64)) \
  --model.denoiser.init_args.num_blocks $NB --model.denoiser.init_args.num_text_blocks $NT \
  --data.pred_dataset.init_args.jsonl_path $DATA/gpic_test_50k.jsonl \
  --data.train_dataset.init_args.tar_dir $DATA/train/ --data.val_dataset.init_args.tar_dir $DATA/val/ \
  --torch_hub_dir $WC/.cache/torch/hub --huggingface_cache_dir $WC/cache/huggingface/hub > $OUT/predict.log 2>&1 \
  || { echo "predict failed (see $OUT/predict.log)"; exit 4; }
NPZ=$(find $OUT/pred -name "output.np[yz]" | head -1)
[ -n "$NPZ" ] || { echo "no samples (see $OUT/predict.log)"; exit 4; }
t1=$(date +%s)
python -m torch.distributed.run --standalone --nproc_per_node 8 $HERE/tools/fast_eval.py "$NPZ" $DATA/reference_stats/test_stats.npz \
  $OUT/metrics.json --workers 6 > $OUT/eval.log 2>&1 || { echo "DINOv2 scoring failed (see $OUT/eval.log)"; exit 5; }
t2=$(date +%s)
if [ -n "${GPIC_TF_PYTHON:-}" ]; then   # the official Inception metrics: TF features in 8 shards, one per GPU, joined in image order
  IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  # gpic_eval keeps the Inception graph (classify_image_graph_def.pb, downloaded on first use) under XDG_CACHE_HOME
  export XDG_CACHE_HOME=${GPIC_XDG_CACHE:-$WC/cache/gpic_eval_xdg} TF_FORCE_GPU_ALLOW_GROWTH=true
  pids=()
  for s in $(seq 0 $((${#G[@]} - 1))); do
    CUDA_VISIBLE_DEVICES=${G[$s]} $GPIC_TF_PYTHON $HERE/tools/official_inception_sharded.py features "$NPZ" $s ${#G[@]} $OUT/inception_f$s.npy \
      > $OUT/inception_shard$s.log 2>&1 & pids+=($!)
  done
  ok=1; for p in "${pids[@]}"; do wait $p || ok=0; done
  [ $ok = 1 ] && $GPIC_TF_PYTHON $HERE/tools/official_inception_sharded.py metrics $DATA/reference_stats/test_stats.npz $OUT/metrics_inception.json \
    $(for s in $(seq 0 $((${#G[@]} - 1))); do echo $OUT/inception_f$s.npy; done) > $OUT/inception.log 2>&1 \
    || echo "Inception metrics failed (see $OUT/inception*.log)"
  rm -f $OUT/inception_f*.npy
fi
rm -f "$NPZ"   # 50K samples, 9.8 GB; the metrics stay
echo "{\"predict_s\": $((t1 - t0)), \"dinov2_s\": $((t2 - t1)), \"total_s\": $(($(date +%s) - t0))}" > $OUT/phase_times.json
cat $OUT/metrics.json
