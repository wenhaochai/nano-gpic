#!/bin/bash
#SBATCH -J gpic-mfu-fp8pairs
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Native rowwise FP8, phase 1 (GPIC_MFU_FP8=native, src/models/layers/fp8_native.py) against bf16, both on E + packed K2,
# as pairs on the two GPU halves: GPUs 0-3 run the bf16 arms and GPUs 4-7 the FP8 arms of the same widths, 4 GPUs x 128
# each (global batch 512; d1792 64 x 2), the same batches, so the pair compares like for like. Half B starts 150 s later,
# so each half's cold compile (GPUs idle) overlaps the other half's training (the controller, 2026-11-07: multi-arm jobs
# left all 8 GPUs at 0% during their compiles). The FP8 and packed K2 tests on GPU 0 first.
# usage: fp8n_pairs.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/fp8pairs_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
export PIXGEN_SRC=$WT/baselines/PixelGen MFU_WT=$WT GPIC_MFU_DET_DATA=1 GPIC_SHARD_SEED=20261106 MFU_LIMIT=1500
export GPIC_MFU_ATTN=fa3 GPIC_MFU_GPU_RESIZE=1 GPIC_MFU_UINT8=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_MFU_PACK=1
cd $WT
for t in test_fp8_native test_fused_qk_packed; do CUDA_VISIBLE_DEVICES=0 python tools/mfu/tests/$t.py 2>&1 | tail -4 | tee -a $OUT/summary.txt; done
grep -q "FUSED_QK_PACKED_TEST PASS" $OUT/summary.txt && grep -q "FP8_NATIVE_TEST PASS" $OUT/summary.txt || { echo "a GPU test failed"; exit 1; }
MUON=(--config configs_h100/muon_ladder.yaml --trainer.devices 4)
FS4='{"class_path":"src.utils.fsdp_setup.NoSyncModelParallelStrategy","init_args":{"data_parallel_size":4,"tensor_parallel_size":1,"save_distributed_checkpoint":false}}'
arm() { bash $WT/jobs/mfu/arm.sh $OUT "$@"; local rc=$?; echo "[$(date +%T)] $1 rc=$rc" | tee -a $OUT/summary.txt; }
SPECS=("640 9 2 2048 ddp 128" "768 10 2 2048 ddp 128" "896 11 2 1536 ddp 128" "1024 12 3 1024 ddp 128"
       "1280 14 3 1024 fsdp7 128" "1536 16 4 768 fsdp 128" "1792 18 4 768 fsdp 64")
half() {   # <gpus> <master port> <FP8: 0|1> <tag suffix>
  local spec d nb nt steps mode mb
  for spec in "${SPECS[@]}"; do
    read -r d nb nt steps mode mb <<< "$spec"
    (
      export CUDA_VISIBLE_DEVICES=$1 MASTER_PORT=$2 MFU_MICRO=$mb
      [ "$3" = 1 ] && export GPIC_MFU_FP8=native
      case $mode in
        fsdp7) export GPIC_MFU_FSDP=1 GPIC_MFU_FSDP_GROUP=7 MFU_STRATEGY=$FS4 ;;
        fsdp) export GPIC_MFU_FSDP=1 MFU_STRATEGY=$FS4 ;;
      esac
      arm d${d}_${mode}_m${mb}_$4 $d $nb $nt $steps "${MUON[@]}" --data.train_dataset.init_args.group_window $mb
    )
  done
}
half 0,1,2,3 29511 0 EK4 > $OUT/halfA.log 2>&1 &
sleep 150
half 4,5,6,7 29512 1 EKN4 > $OUT/halfB.log 2>&1 &
wait
python - $OUT <<'PY' | tee -a $OUT/summary.txt
import json, glob, math, os, statistics as st, sys
out = sys.argv[1]
GF = {640: 104.1, 768: 163.8, 896: 242.1, 1024: 346.9, 1280: 622.4, 1536: 1024.9, 1792: 1554.8}
def load(tag):
    f = os.path.join(out, tag, "bench", "steps_rank0.json")
    return json.load(open(f))["rows"] if os.path.exists(f) else None
for d in sorted(glob.glob(os.path.join(out, "d*_EK4")), key=lambda p: int(os.path.basename(p).split("_")[0][1:])):
    tag = os.path.basename(d); b, f = load(tag), load(tag.replace("_EK4", "_EKN4"))
    if not b or not f:
        print(f"{tag}: missing rows (bf16 {bool(b)}, FP8 {bool(f)})"); continue
    acc = 2 if "_m64_" in tag else 1
    tm = lambda r: acc * st.mean(x.get("step_ms", 1000 * (x["t_end"] - x["t_start"])) for x in r[len(r) // 2:])
    n = min(len(b), len(f)); lb = st.mean(x["loss"] for x in b[n - 200:n]); lf = st.mean(x["loss"] for x in f[n - 200:n])
    nf = sum(1 for r in (b, f) for x in r if not math.isfinite(x["loss"]))
    w = int(tag.split("_")[0][1:])
    print(f"{tag[:-4]}: bf16 {tm(b):.1f} ms, FP8 {tm(f):.1f} ms ({tm(f) / tm(b) - 1:+.1%}); loss over the last 200 steps bf16 {lb:.5f}, "
          f"FP8 {lf:.5f}; non-finite losses {nf}")
PY
echo "DONE $OUT"
