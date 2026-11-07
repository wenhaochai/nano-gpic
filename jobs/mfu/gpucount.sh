#!/bin/bash
#SBATCH -J gpic-mfu-gpucount
#SBATCH -A pli
#SBATCH -p pli-c
#SBATCH -q pli-short
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --time=01:30:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
# The GPU-count table of the new ladders (genmodel / owner 2026-11-07: the best GPU count per width at global batch
# 1024): the production path (jobs/mfu/arm.sh's settings: CompileWarmup, HostTuning, the C++ launcher, tcmalloc, region
# crop, the step timer, no checkpoints) on SLURM_NNODES nodes x 8 GPUs, Lightning's SLURM launch (one task per GPU),
# micro-batch 32, accumulation 1024 / (32 x GPUs), Muon overlay, 256 optimizer steps per arm.
# usage: sbatch --nodes <N> gpucount.sh <code worktree> <d>:<mode> ...   mode: ddp | fsdp (FSDP2 no-sync; levers in
#   GPUCOUNT_ENV, default A C D K K2; fsdp adds GPIC_MFU_FSDP=1 and NoSyncModelParallelStrategy)
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}; shift
NN=${SLURM_NNODES:-1}; NG=$((NN * 8)); ACC=$((1024 / (32 * NG)))
OUT=$WC/runs/gpic-speed/mfu/gpucount_${NG}gpu_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
export PIXGEN_SRC=$WT/baselines/PixelGen GPIC_MFU_DET_DATA=1 GPIC_SHARD_SEED=20261106
export GPIC_MFU_ATTN=fa3 GPIC_MFU_GPU_RESIZE=1 GPIC_MFU_UINT8=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 ${GPUCOUNT_ENV:-}
export GPIC_REGION_CROP=1 TORCHINDUCTOR_CPP_WRAPPER=1 WANDB_MODE=disabled
[ -f /usr/lib64/libtcmalloc.so.4 ] && export LD_PRELOAD=/usr/lib64/libtcmalloc.so.4 TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=100000000000
CB='[{"class_path":"src.callbacks.compile_warmup.CompileWarmup"},{"class_path":"src.callbacks.host_tuning.HostTuning"},{"class_path":"src.callbacks.bench.BenchTimer"}]'
geom() { case $1 in 640) echo "9 2" ;; 768) echo "10 2" ;; 896) echo "11 2" ;; 1024) echo "12 3" ;; 1280) echo "14 3" ;; 1536) echo "16 4" ;; 1792) echo "18 4" ;; esac; }
for spec in "$@"; do
  d=${spec%%:*}; mode=${spec#*:}; read -r nb nt <<< "$(geom $d)"
  TAG=d${d}_${mode}_${NG}gpu; mkdir -p $OUT/$TAG
  STRAT=(); ENVX=""
  if [ $mode = fsdp ]; then
    ENVX="GPIC_MFU_FSDP=1"
    STRAT=(--trainer.strategy "{\"class_path\":\"src.utils.fsdp_setup.NoSyncModelParallelStrategy\",\"init_args\":{\"data_parallel_size\":$NG,\"tensor_parallel_size\":1,\"save_distributed_checkpoint\":false}}")
  fi
  echo "[$(date +%T)] start $TAG nodes=$NN gpus=$NG accumulation=$ACC" | tee -a $OUT/summary.txt
  ( export BENCH_OUT=$OUT/$TAG/bench GPIC_SHARD_LOG_DIR=$OUT/$TAG $ENVX
    cd $WT/baselines/PixelGen && export PYTHONPATH=$PWD
    timeout -k 60 1500 srun --kill-on-bad-exit=1 --ntasks-per-node=8 python main.py fit --config configs_h100/lite_ladder.yaml \
      --config configs_h100/muon_ladder.yaml --trainer.default_root_dir $OUT/$TAG --trainer.max_steps 256 \
      --trainer.num_nodes $NN --trainer.devices 8 --trainer.accumulate_grad_batches $ACC --trainer.callbacks "$CB" \
      --trainer.enable_checkpointing false --model.denoiser.init_args.hidden_size $d --model.denoiser.init_args.num_groups $((d / 64)) \
      --model.denoiser.init_args.num_blocks $nb --model.denoiser.init_args.num_text_blocks $nt \
      --tags.exp mfu_$TAG --trainer.logger.init_args.name mfu_$TAG "${STRAT[@]}" > $OUT/$TAG/train.log 2>&1 )
  echo "[$(date +%T)] $TAG rc=$?" | tee -a $OUT/summary.txt
done
python - $OUT $ACC <<'PY' | tee -a $OUT/summary.txt
import json, os, glob, statistics as st, sys
out, acc = sys.argv[1], int(sys.argv[2])
GF = {640: 104.1, 768: 163.8, 896: 242.1, 1024: 346.9, 1280: 622.4, 1536: 1024.9, 1792: 1554.8}
for d in sorted(glob.glob(os.path.join(out, "d*"))):
    f = os.path.join(d, "bench", "steps_rank0.json")
    if not os.path.exists(f):
        print(os.path.basename(d), "no timer rows"); continue
    rows = json.load(open(f))["rows"]
    ms = [r.get("step_ms", 1000 * (r["t_end"] - r["t_start"])) for r in rows][len(rows) // 2:]
    w = int(os.path.basename(d).split("_")[0][1:])
    mb = st.mean(ms)
    print(f"{os.path.basename(d)}: micro-batch {mb:.2f} ms mean (second half), optimizer step {acc * mb:.1f} ms, "
          f"MFU {GF[w] * 1e9 * 32 / (mb / 1e3) / 989.4e12:.1%}")
PY
echo "DONE $OUT"
