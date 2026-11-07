#!/bin/bash
#SBATCH -J gpic-mfu-muon
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Muon on the DDP path (docs/mfu.md "Muon"; configs_h100/muon_ladder.yaml: MuP scale, split parts, AdamW for the rest,
# warmup 1000, global batch 1024 = 8 GPUs x 32 x accumulation 4, EMA 0.9996 once per optimizer step):
# 1. tools/mfu/tests/test_muon_reference.py on one GPU and on two (torchrun);
# 2. production path (jobs/mfu/arm.sh, levers A C D, DDP), 256 optimizer steps (1024 micro-batches): d640 Muon; d640
#    Muon to step 128 with its last.ckpt, then a resume from it to 256 (the loss must continue); d640 AdamW (the same
#    accumulation and EMA) for the step cost; d1024 Muon and AdamW the same way.
# usage: muon_smoke.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/muon_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
export PIXGEN_SRC=$WT/baselines/PixelGen MFU_WT=$WT GPIC_MFU_DET_DATA=1 GPIC_SHARD_SEED=20261106 MFU_LIMIT=1200
export GPIC_MFU_ATTN=fa3 GPIC_MFU_GPU_RESIZE=1 GPIC_MFU_UINT8=1
cd $WT/baselines/PixelGen
( CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD timeout 600 python $WT/tools/mfu/tests/test_muon_reference.py > $OUT/test_1gpu.log 2>&1
  echo "[$(date +%T)] reference test, 1 GPU: $(tail -1 $OUT/test_1gpu.log)" >> $OUT/summary.txt ) &
( CUDA_VISIBLE_DEVICES=1,2 PYTHONPATH=$PWD timeout 600 torchrun --standalone --nproc_per_node 2 $WT/tools/mfu/tests/test_muon_reference.py > $OUT/test_2gpu.log 2>&1
  echo "[$(date +%T)] reference test, 2 GPUs: $(grep -h 'ALL OK\|SOME CHECKS' $OUT/test_2gpu.log | tail -1)" >> $OUT/summary.txt ) &
wait
cd $WT
MUON=(--config configs_h100/muon_ladder.yaml)
ADAMW=(--trainer.accumulate_grad_batches 4 --model.ema_tracker.init_args.decay 0.9996)
arm() { bash $WT/jobs/mfu/arm.sh $OUT "$@"; echo "[$(date +%T)] $1 rc=$?" | tee -a $OUT/summary.txt; }
L640=(--model.optimizer.init_args.lr 6.4e-4 --model.optimizer.init_args.muon_lr 1.7e-3)
arm d640_muon 640 9 2 256 "${MUON[@]}" "${L640[@]}"
MFU_SAVE_EVERY=128 MFU_KEEP_CKPT=1 arm d640_muon_to128 640 9 2 128 "${MUON[@]}" "${L640[@]}"
CK=$(ls $OUT/d640_muon_to128/exp_mfu_d640_muon_to128/last*.ckpt $OUT/d640_muon_to128/last*.ckpt 2>/dev/null | head -1)
echo "resume from: $CK" | tee -a $OUT/summary.txt
MFU_RESUME=$CK arm d640_muon_resumed 640 9 2 256 "${MUON[@]}" "${L640[@]}"
arm d640_adamw 640 9 2 256 "${ADAMW[@]}" --model.optimizer.init_args.lr 6.4e-4
arm d1024_muon 1024 12 3 256 "${MUON[@]}" --model.optimizer.init_args.lr 3.3e-4 --model.optimizer.init_args.muon_lr 1.7e-3
arm d1024_adamw 1024 12 3 256 "${ADAMW[@]}" --model.optimizer.init_args.lr 3.3e-4
python - $OUT <<'PY' | tee -a $OUT/summary.txt
import json, os, statistics as st, sys
out = sys.argv[1]
def rows(tag):
    f = os.path.join(out, tag, "bench", "steps_rank0.json")
    return json.load(open(f))["rows"] if os.path.exists(f) else []
for tag in ("d640_muon", "d640_muon_to128", "d640_muon_resumed", "d640_adamw", "d1024_muon", "d1024_adamw"):
    r = rows(tag)
    if not r:
        print(f"{tag}: no timer rows"); continue
    ms = [x.get("step_ms", 1000 * (x["t_end"] - x["t_start"])) for x in r]
    opt = [sum(ms[i:i + 4]) for i in range(len(ms) // 2 // 4 * 4, len(ms) - 3, 4)]   # second half, per optimizer step
    loss = [x["loss"] for x in r if x.get("loss") is not None]
    print(f"{tag}: {len(r)} micro-batches, optimizer step {st.median(opt):.1f} ms median ({st.mean(opt):.1f} mean) over the "
          f"second half; loss first 64 {st.mean(loss[:64]):.4f}, last 64 {st.mean(loss[-64:]):.4f}")
PY
grep -h "MuonAdamW\]\|Traceback\|Error" $OUT/*/train.log | sort | uniq -c | head -8
echo "DONE $OUT"
