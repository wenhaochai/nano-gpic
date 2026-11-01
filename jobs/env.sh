# Common environment for GPIC jobs on H100 compute nodes (sourced by job scripts).
export WC=/scratch/gpfs/GROUP/USER
export REPO=$WC/project/gpic
export PIXGEN=$REPO/baselines/PixelGen
export JOBID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
export TMPDIR=$WC/tmp/job/$JOBID
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

module load proxy/default 2>/dev/null || true
source $WC/envs/pixgen/bin/activate

export HF_HOME=$WC/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=$WC/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=$WC/cache/inductor/gpic
export TRITON_CACHE_DIR=$WC/cache/triton/gpic
export WANDB_DIR=$WC/runs/gpic-speed/wandb
export WANDB_PROJECT=gpic-speed
mkdir -p "$WANDB_DIR"
export PYTHONPATH=$PIXGEN
export PYTHONUNBUFFERED=1

echo "[env] job=$JOBID host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-all} $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
