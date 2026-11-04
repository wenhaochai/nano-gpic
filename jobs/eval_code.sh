#!/bin/bash
#SBATCH -J gpic-code-eval
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:20:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# GPIC evaluation of one plan A checkpoint at each number of network calls: jobs/eval_ladder.sh's protocol (50K test captions, EMA net, CFG 1.0, the official noise, FD-DINOv2 / PRDC / MMD),
# with configs_h100/code_sample.yaml. 50 calls is the main number.
# usage: eval_code.sh <size> <d> <nb> <nt> <step> <calls...>     e.g. eval_code.sh d640 640 9 2 66176 50
# Env (smoke tests only): LADDER_WT, LADDER_RUN_ROOT.
# Env: EVAL_NOISE=prompt gives each prompt its own x_T and writes eval_<step>_calls<N>_prompt. The default is the official x_T: one tensor for all prompts. z0 is per prompt in both cases (lightning_model_gpic.py).
set -eo pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
NAME=$1 D=$2 NB=$3 NT=$4 STEP=$(printf %06d $5); shift 5
WT=${LADDER_WT:-$WC/project/gpic-wt/code_run}
RUN=${LADDER_RUN_ROOT:-$WC/runs/gpic-speed}/exp_ladder_$NAME
CKPT=$(ls -t $RUN/ms-*$STEP*.ckpt 2>/dev/null | head -1); [ -n "$CKPT" ] || { echo "no ms checkpoint for step $STEP in $RUN"; exit 2; }
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD
# the checkpoint loads with strict=False, so wrong sizes would give a wrong FD and no error
python - "$CKPT" $D $NB $NT <<'PY'
import re, sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)["state_dict"]
d, nb, nt = map(int, sys.argv[2:])
n = lambda part: len({m.group(1) for k in sd for m in [re.match(rf"ema_denoiser\.{part}\.(\d+)\.", k)] if m})
got = (sd["ema_denoiser.y_pos_embedding"].shape[-1], n("blocks"), n("text_refine_blocks"))
assert got == (d, nb, nt), f"checkpoint has d, nb, nt = {got}, arguments {(d, nb, nt)}"
assert "ema_denoiser.code_prior.out.weight" in sd, "this checkpoint is not plan A"
PY
SUF= NOISE=
if [ "${EVAL_NOISE:-official}" = prompt ]; then SUF=_prompt NOISE="--data.pred_dataset.init_args.noise_per_prompt true"; fi
for CALLS in "$@"; do
  OUT=$RUN/eval_${STEP}_calls$CALLS$SUF; mkdir -p $OUT; echo "ckpt=$CKPT code=$(git -C $WT rev-parse HEAD) calls=$CALLS" | tee $OUT/ckpt.txt
  t0=$(date +%s)
  WANDB_MODE=disabled python main.py predict --config configs_h100/lite_ladder_sample.yaml --config configs_h100/code_sample.yaml --trainer.devices 8 \
    --trainer.default_root_dir "$OUT/pred" --ckpt_path "$CKPT" --model.diffusion_sampler.init_args.num_steps $CALLS \
    --model.denoiser.init_args.hidden_size $D --model.denoiser.init_args.num_groups $((D / 64)) \
    --model.denoiser.init_args.num_blocks $NB --model.denoiser.init_args.num_text_blocks $NT $NOISE > $OUT/predict.log 2>&1
  t1=$(date +%s)
  NPZ=$(find "$OUT/pred" -name output.npz | head -1)
  python -m torch.distributed.run --standalone --nproc_per_node 8 $WT/tools/fast_eval.py "$NPZ" \
    $WC/data/gpic/reference_stats/test_stats.npz $OUT/metrics.json --workers 11 > $OUT/eval.log 2>&1
  t2=$(date +%s)
  echo "{\"predict_s\": $((t1-t0)), \"eval_s\": $((t2-t1))}" | tee $OUT/phase_times.json
  cat $OUT/metrics.json
  rm -f "$NPZ"   # 50K x 196 KB; metrics.json keeps the result
done
