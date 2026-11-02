#!/bin/bash
# 2-GPU Lightning smoke of the ladder training path on vis2 (two A40s; never sources jobs/env.sh): a tiny
# denoiser (d256, 2+1 blocks) with lite_ladder.yaml and the same command-line overrides as jobs/train_ladder.sh,
# 40 steps (last.ckpt every 10, ms every 20), then a resume from last.ckpt to step 60. The text encoder uses sdpa (the
# A40 cannot run its FlexAttention kernel). TORCH_LOGS=recompiles lists every recompilation.
# usage (on vis2): bash tools/smoke_ladder_vis.sh <out_dir>
set -o pipefail
OUT=$1; mkdir -p $OUT/tmp
WC=/scratch/gpfs/GROUP/USER
WT=$(cd "$(dirname "$0")/.." && pwd)
source $WC/envs/pixgen/bin/activate
export HF_HOME=$WC/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=$WC/.cache/torch
export TMPDIR=$OUT/tmp TORCHINDUCTOR_CACHE_DIR=$OUT/inductor TRITON_CACHE_DIR=$OUT/triton
export WANDB_MODE=offline WANDB_DIR=$OUT PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 TORCH_LOGS=recompiles
export GPIC_SHARD_SEED=1762100000
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD
CB='[{"class_path":"src.callbacks.model_checkpoint.CheckpointHook","init_args":{"filename":"resume-{step:06d}","monitor":null,"save_last":true,"save_top_k":0,"every_n_train_steps":10}},{"class_path":"src.callbacks.model_checkpoint.CheckpointHook","init_args":{"filename":"ms-{step:06d}","monitor":null,"save_last":false,"save_top_k":-1,"every_n_train_steps":20}},{"class_path":"src.callbacks.compile_warmup.CompileWarmup"}]'
COMMON=(--config configs_h100/lite_ladder.yaml --trainer.devices 2 --trainer.default_root_dir $OUT
  --trainer.callbacks "$CB" --trainer.log_every_n_steps 5
  --model.denoiser.init_args.hidden_size 256 --model.denoiser.init_args.num_groups 4
  --model.denoiser.init_args.num_blocks 2 --model.denoiser.init_args.num_text_blocks 1
  --data.train_dataset.init_args.tar_pattern 'gpic_train_{00000..00007}.tar' --data.train_num_workers 2
  --tags.exp ladder_smoke --trainer.logger.init_args.name ladder_smoke
  --trainer.logger.init_args.id ladder-smoke-vis-$(date +%m%d%H%M) --trainer.logger.init_args.offline true)
python main.py fit "${COMMON[@]}" --trainer.max_steps 40 > $OUT/run1.log 2>&1
echo "run1 rc=$?"
python main.py fit "${COMMON[@]}" --trainer.max_steps 60 --ckpt_path $OUT/exp_ladder_smoke/last.ckpt > $OUT/run2.log 2>&1
echo "run2 rc=$?"
ls -la $OUT/exp_ladder_smoke/
