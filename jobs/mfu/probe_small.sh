#!/bin/bash
#SBATCH -J gpic-mfu-probe-small
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# P-train of the optimization scan (gpic-28 2026-11-09; owner popup 「两个都跑」): the production sweep path at d9 and d10, where each
# micro-batch waits 20-31 ms in the shared bucket's gloo all-reduce (x-d9 72.5 ms per step, x-d10 76.2 ms, x-d11 101.2 ms with 1.6 ms).
# Code: mfu-probe-small = the sweep code noema_cap_run 8e1b9ef + X per bucket (sweep-cgbucket 3878f13) + S2 (mfu 9b28f47, the pulls and
# decisions k batches ahead in a thread). Each arm: train_ladder.sh with train_sweep_noema_cap_nccl.sh's settings (NCCL 2.27.7, K, K2,
# NS, MA, shared bucket, 128 x 1, Muon), 2048 steps, one shard seed, no last.ckpt, the step timer; then the arm's checkpoints are removed.
# Arms: d9 base, S2, 16 data workers, X, S2 + X; d10 base, S2, X. The summary: mean and median step of steps 1024-2047, data wait,
# the shared bucket's all-reduce wait. S2's thread can abort the process at exit (no clean stop yet): an arm's rc is reported, the
# timer rows are written before exit.
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
WT=$WC/project/gpic-wt/probe_small_run
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
OUT=$WC/runs/gpic-speed/mfu_probe/small_$(date +%m%d_%H%M)
mkdir -p $OUT
export GPIC_MFU_NO_EMA=1 LD_PRELOAD=$WC/envs/nccl/2.27.7/libnccl.so.2
S=2048 GB=1024 MB=128 MLR=1.360e-02 ALR=0.00064 WDK=0.002752 CDP=20 PAT='gpic_train_{00000..07999}.tar'
WU=$((S / 100)); CD=$(( (S * CDP / 100) / 32 * 32 ))
arm() {   # <tag> <d> <nb> <nt> [extra main.py args]; env toggles set by the caller
  local tag=$1 d=$2 nb=$3 nt=$4; shift 4
  local root=$OUT/$tag size=d$2
  mkdir -p $root/exp_ladder_$size; echo 20261106 > $root/exp_ladder_$size/shard_seed.txt
  ( export GPIC_MFU_FSDP=0 GPIC_MFU_FSDP_GROUP=1 GPIC_MFU_FSDP_NOSYNC=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_MFU_PATCHIFY=0 \
      GPIC_MFU_NS_FUSED=1 GPIC_MFU_LOCAL_OPT=0 GPIC_MFU_MUON_ADD=1
    export LADDER_WT=$WT LADDER_RUN_ROOT=$root LADDER_LAST_EVERY=100000 BENCH_OUT=$root/bench LADDER_PARTIAL_OK=1
    echo "[$(date +%T)] arm $tag: d=$d nb=$nb nt=$nt env: $(env | grep -E '^(GPIC_SHARED_BUCKET_AHEAD|GPIC_MFU_CUDAGRAPH|GPIC_PROF_DEN_COMPILE_MODE|TORCHINDUCTOR_CPP_WRAPPER)=' | tr '\n' ' ') args: $*"
    timeout -k 60 900 bash $WT/jobs/train_ladder.sh $size $d $nb $nt $S \
      --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr $ALR --model.optimizer.init_args.muon_lr $MLR \
      --model.optimizer.init_args.weight_decay 0 --model.optimizer.init_args.muon_weight_decay $WDK \
      --model.lr_scheduler.init_args.num_warmup_steps $WU --model.lr_scheduler.init_args.num_cooldown_steps $CD \
      --model.lr_scheduler.init_args.max_steps $S --model.lr_scheduler.init_args.min_lr 3e-5 \
      --data.train_batch_size $MB --trainer.accumulate_grad_batches $((GB / 8 / MB)) --data.train_dataset.init_args.shared_bucket true \
      --data.train_dataset.init_args.tar_pattern "$PAT" --trainer.logger.init_args.name probe_${tag} --trainer.logger.init_args.id probe-$tag-$(date +%s) \
      "$@" > $root/train.log 2>&1
    echo "[$(date +%T)] arm $tag rc=$?" )
  find $root -name "*.ckpt" -delete
}
XENV="GPIC_MFU_CUDAGRAPH=1 GPIC_PROF_DEN_COMPILE_MODE=reduce-overhead TORCHINDUCTOR_CPP_WRAPPER=0"
arm d9-base 640 9 2
GPIC_SHARED_BUCKET_AHEAD=2 arm d9-s2 640 9 2
arm d9-w16 640 9 2 --data.train_num_workers 16
env $XENV bash -c "$(declare -f arm); $(declare -p OUT WT WC S GB MB MLR ALR WDK CDP PAT WU CD); arm d9-x 640 9 2"
env $XENV GPIC_SHARED_BUCKET_AHEAD=2 bash -c "$(declare -f arm); $(declare -p OUT WT WC S GB MB MLR ALR WDK CDP PAT WU CD); arm d9-s2x 640 9 2"
arm d10-base 768 10 2
GPIC_SHARED_BUCKET_AHEAD=2 arm d10-s2 768 10 2
env $XENV bash -c "$(declare -f arm); $(declare -p OUT WT WC S GB MB MLR ALR WDK CDP PAT WU CD); arm d10-x 768 10 2"
source $WC/envs/pixgen/bin/activate
python - $OUT <<'PY'
import glob, json, os, re, sys, statistics as stt
out = sys.argv[1]
print(f"{'arm':10s} {'mean ms':>8s} {'median':>7s} {'max-rank mean':>13s} {'data wait':>9s} {'all-reduce':>10s} {'steps':>5s}")
base = {}
for root in sorted(glob.glob(f"{out}/*/")):
    tag = os.path.basename(root.rstrip("/"))
    files = sorted(glob.glob(f"{root}bench/steps_rank*.json"))
    if not files:
        print(f"{tag:10s} no timer rows"); continue
    per_rank = []
    for f in files:
        rows = [r for r in json.load(open(f))["rows"] if 1024 <= r["step"] < 2048 and r.get("step_ms")]
        per_rank.append(rows)
    r0 = per_rank[0]
    ms = [r["step_ms"] for r in r0]
    dw = [1000 * r.get("data_wait_s", 0) for r in r0]
    mx = max(stt.mean(r["step_ms"] for r in rr) for rr in per_rank if rr)
    log = open(f"{root}train.log", errors="replace").read()
    ar = re.findall(r"all-reduce ([0-9.]+) ms", log)
    print(f"{tag:10s} {stt.mean(ms):8.1f} {stt.median(ms):7.1f} {mx:13.1f} {stt.mean(dw):9.1f} {(ar[-1] if ar else '-'):>10s} {len(ms):5d}")
PY
echo PROBE_SMALL_DONE $OUT
