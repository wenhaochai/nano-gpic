#!/bin/bash
#SBATCH -J gpic-mfu-pairs
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Two settings as pairs on the two GPU halves, in the ladder's setting (shared bucket, Muon, A C D K K2, 128 per GPU; d1792
# 64 x 2): GPUs 0-3 run setting A and GPUs 4-7 setting B at the same widths, 4 GPUs each (global batch 512), the same
# batches, so each pair compares like for like. Half B starts 150 s later, so each half's cold compile (GPUs idle)
# overlaps the other half's training. The named tests run on GPU 0 first and must print "<NAME>_TEST PASS". Everything
# comes in as arguments: wq runs a job in a clean environment.
# usage: pairs.sh <code worktree> <name> <env of A or none> <env of B or none> <tests or none> <width spec>...
#   env: space-separated VAR=value pairs, e.g. "GPIC_MFU_PATCHIFY=1"; tests: space-separated scripts in tools/mfu/tests,
#   e.g. "test_patchify"; width spec: "<d> <steps>", d one of SPECS below.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree} NAME=${2:?name} EA=${3:?env of A} EB=${4:?env of B} TESTS=${5:?tests or none}
shift 5
WIDTHS=("$@")
OUT=$WC/runs/gpic-speed/mfu/pairs_${NAME}_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
export PIXGEN_SRC=$WT/baselines/PixelGen MFU_WT=$WT GPIC_MFU_DET_DATA=1 GPIC_SHARD_SEED=20261106 MFU_LIMIT=1500
export GPIC_MFU_ATTN=fa3 GPIC_MFU_GPU_RESIZE=1 GPIC_MFU_UINT8=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_SHARED_BUCKET=1
cd $WT
[ "$TESTS" = none ] && TESTS=
for t in $TESTS; do
  CUDA_VISIBLE_DEVICES=0 python tools/mfu/tests/$t.py 2>&1 | grep -v -i warn | tail -8 | tee -a $OUT/summary.txt
  grep -q "_TEST PASS" <(tail -1 $OUT/summary.txt) || { echo "test $t failed"; exit 1; }
done
MUON=(--config configs_h100/muon_ladder.yaml --trainer.devices 4)
FS4='{"class_path":"src.utils.fsdp_setup.NoSyncModelParallelStrategy","init_args":{"data_parallel_size":4,"tensor_parallel_size":1,"save_distributed_checkpoint":false}}'
declare -A SPECS=([640]="9 2 ddp 128" [768]="10 2 ddp 128" [896]="11 2 ddp 128" [1024]="12 3 ddp 128"
                  [1280]="14 3 fsdp7 128" [1536]="16 4 fsdp 128" [1792]="18 4 fsdp 64")
arm() { bash $WT/jobs/mfu/arm.sh $OUT "$@"; local rc=$?; echo "[$(date +%T)] $1 rc=$rc" | tee -a $OUT/summary.txt; }
half() {   # <gpus> <master port> <env or none> <tag suffix>
  local w d steps nb nt mode mb
  for w in "${WIDTHS[@]}"; do
    read -r d steps <<< "$w"
    read -r nb nt mode mb <<< "${SPECS[$d]}"
    (
      export CUDA_VISIBLE_DEVICES=$1 MASTER_PORT=$2 MFU_MICRO=$mb
      [ "$3" != none ] && export $3
      case $mode in
        fsdp7) export GPIC_MFU_FSDP=1 GPIC_MFU_FSDP_GROUP=7 MFU_STRATEGY=$FS4 ;;
        fsdp) export GPIC_MFU_FSDP=1 MFU_STRATEGY=$FS4 ;;
      esac
      arm d${d}_${mode}_m${mb}_$4 $d $nb $nt $steps "${MUON[@]}"
    )
  done
}
half 0,1,2,3 29521 "$EA" A > $OUT/halfA.log 2>&1 &
sleep 150
half 4,5,6,7 29522 "$EB" B > $OUT/halfB.log 2>&1 &
wait
echo "half A: $EA; half B: $EB" | tee -a $OUT/summary.txt
python - $OUT <<'PY' | tee -a $OUT/summary.txt
import json, glob, math, os, statistics as st, sys
out = sys.argv[1]
def load(tag):
    f = os.path.join(out, tag, "bench", "steps_rank0.json")
    return json.load(open(f))["rows"] if os.path.exists(f) else None
for d in sorted(glob.glob(os.path.join(out, "d*_A")), key=lambda p: int(os.path.basename(p).split("_")[0][1:])):
    tag = os.path.basename(d); a, b = load(tag), load(tag[:-2] + "_B")
    if not a or not b:
        print(f"{tag[:-2]}: missing rows (A {bool(a)}, B {bool(b)})"); continue
    acc = 2 if "_m64" in tag else 1
    tm = lambda r: acc * st.mean(x.get("step_ms", 1000 * (x["t_end"] - x["t_start"])) for x in r[len(r) // 2:])
    md = lambda r: acc * st.median(x.get("step_ms", 1000 * (x["t_end"] - x["t_start"])) for x in r[len(r) // 2:])
    n = min(len(a), len(b)); la = st.mean(x["loss"] for x in a[n - 200:n]); lb = st.mean(x["loss"] for x in b[n - 200:n])
    nf = sum(1 for r in (a, b) for x in r if not math.isfinite(x["loss"]))
    first = next((i for i in range(n) if a[i]["loss"] != b[i]["loss"]), None)
    print(f"{tag[:-2]}: mean step A {tm(a):.1f} ms, B {tm(b):.1f} ms ({tm(b) / tm(a) - 1:+.1%}); median A {md(a):.1f}, B {md(b):.1f}; "
          f"loss over the last 200 steps A {la:.5f}, B {lb:.5f}; losses equal up to step {n if first is None else first} of {n}; "
          f"non-finite losses {nf}")
PY
echo "DONE $OUT"
