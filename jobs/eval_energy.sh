#!/bin/bash
#SBATCH -J gpic-energy-eval
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# GPIC evaluation of one energy-head checkpoint at each number of network calls: the protocol of jobs/eval_ladder.sh with each prompt's own noise (50K test captions, EMA net, no CFG, FD-DINOv2 / PRDC / MMD), with configs_h100/energy_sample.yaml. 50 calls is the main number.
# usage: eval_energy.sh <d> <nb> <nt> <step> <calls...>
# Env (smoke tests only): LADDER_WT, LADDER_RUN_ROOT.
set -eo pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
D=$1 NB=$2 NT=$3 STEP=$(printf %06d $4); shift 4
WT=${LADDER_WT:-$WC/project/gpic-wt/energy_real}
RUN=${LADDER_RUN_ROOT:-$WC/runs/gpic-speed/energy_d640}/exp_ladder_d640
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
assert any(k.startswith("ema_denoiser.final_layer.xi_global") for k in sd), "this checkpoint is not the energy head: " + str([k for k in sd if "final_layer" in k][:4])
PY
for CALLS in "$@"; do
  OUT=$RUN/eval_${STEP}_calls${CALLS}_prompt; mkdir -p $OUT; echo "ckpt=$CKPT code=$(git -C $WT rev-parse HEAD) calls=$CALLS" | tee $OUT/ckpt.txt
  t0=$(date +%s)
  WANDB_MODE=disabled python main.py predict --config configs_h100/lite_ladder_sample.yaml --config configs_h100/energy_sample.yaml --trainer.devices 8 \
    --trainer.default_root_dir "$OUT/pred" --ckpt_path "$CKPT" --model.diffusion_sampler.init_args.num_steps $CALLS \
    --model.denoiser.init_args.hidden_size $D --model.denoiser.init_args.num_groups $((D / 64)) \
    --model.denoiser.init_args.num_blocks $NB --model.denoiser.init_args.num_text_blocks $NT \
    --data.pred_dataset.init_args.noise_per_prompt true > $OUT/predict.log 2>&1
  t1=$(date +%s)
  NPZ=$(find "$OUT/pred" -name output.npz | head -1)
  python -m torch.distributed.run --standalone --nproc_per_node 8 $WT/tools/fast_eval.py "$NPZ" \
    $WC/data/gpic/reference_stats/test_stats.npz $OUT/metrics.json --workers 11 > $OUT/eval.log 2>&1
  t2=$(date +%s)
  echo "{\"predict_s\": $((t1-t0)), \"eval_s\": $((t2-t1))}" | tee $OUT/phase_times.json
  cat $OUT/metrics.json
  rm -f "$NPZ"   # 50K x 196 KB; metrics.json keeps the result
done
