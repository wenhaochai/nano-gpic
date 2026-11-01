# Common environment for GPIC jobs on H100 compute nodes (sourced by job scripts).
export WC=/scratch/gpfs/GROUP/USER
export REPO=$WC/project/gpic
export PIXGEN=$REPO/baselines/PixelGen
export JOBID=${FETCHED_JOB_ID:-${SLURM_JOB_ID:-}}
# Per-job TMPDIR, removed at exit only by the shell that created it, and only when it is this job's own directory.
# Without a job id (vis node, login shell) nothing is created or removed: on 2026-11-01 an empty JOBID turned
# TMPDIR into $WC/tmp/job/ and the exit trap deleted every job's temporary directory for 7 minutes.
if [ -n "$JOBID" ] && [ -z "${GPIC_TMP_OWNER:-}" ]; then
  export TMPDIR=$WC/tmp/job/$JOBID
  mkdir -p "$TMPDIR"
  export GPIC_TMP_OWNER=$$ GPIC_TMP_DIR=$TMPDIR
  trap 'if [ "${GPIC_TMP_OWNER:-}" = "$$" ] && [ -n "$JOBID" ] && [ "$GPIC_TMP_DIR" = "$WC/tmp/job/$JOBID" ]; then rm -rf -- "$GPIC_TMP_DIR"; fi' EXIT
fi

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
# wandb >= 0.31 defaults to api.forge.coreweave.com, which proxy does not allow (w826 hung in init, 2026-11-01).
export WANDB_BASE_URL=https://api.wandb.ai
# proxy passes metrics but not file uploads (memory: reference_h100_wandb_proxy_uploads)
export WANDB_IGNORE_GLOBS='*' WANDB_DISABLE_CODE=true WANDB_CONSOLE=off WANDB_X_DISABLE_META=true
mkdir -p "$WANDB_DIR"
export PYTHONPATH=$PIXGEN
export PYTHONUNBUFFERED=1

echo "[env] job=$JOBID host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-all} $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
