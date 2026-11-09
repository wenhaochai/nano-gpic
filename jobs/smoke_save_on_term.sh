#!/bin/bash
#SBATCH -J gpic-smoke-saveterm
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Smoke of the save on the pilot's cut (gpic-28 2026-11-09; owner 「修掉」 on genmodel's bug list): save-on-term-cap = the sweep
# code noema_cap_run 8e1b9ef + src/callbacks/save_on_term.py + data workers that ignore TERM. Per case, the sweep's training path
# (train_sweep_noema_cap_nccl.sh's settings, W&B on) runs under `timeout --kill-after=300 <T>s` as fetch_pending runs a wq guest
# (TERM to the whole process group at T, KILL 300 s later), then again without a limit: it must resume from the step the cut
# saved and finish. Cases: d9 (DDP, 4096 steps, scheduled last.ckpt every 512) and d16 (FSDP2, 1024 steps, every 128).
# Checks: rc 124 and an exit within the grace, one save step for all ranks, the resume's step, the steps and losses after it.
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
WT=$WC/project/gpic-wt/saveterm_cap_run
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
OUT=$WC/runs/gpic-speed/mfu_probe/saveterm_$(date +%m%d_%H%M); mkdir -p $OUT
export GPIC_MFU_NO_EMA=1 LD_PRELOAD=$WC/envs/nccl/2.27.7/libnccl.so.2
run() {   # <tag> <d> <nb> <nt> <steps> <last every> <mb> <fsdp>: one launch of the sweep path
  local tag=$1 d=$2 nb=$3 nt=$4 S=$5 LAST=$6 MB=$7 FS=$8 root=$OUT/$1 size=d$2
  mkdir -p $root/exp_ladder_$size; [ -s $root/exp_ladder_$size/shard_seed.txt ] || echo 20261106 > $root/exp_ladder_$size/shard_seed.txt
  GPIC_MFU_FSDP=$FS GPIC_MFU_FSDP_GROUP=1 GPIC_MFU_FSDP_NOSYNC=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_MFU_PATCHIFY=$FS \
  GPIC_MFU_NS_FUSED=1 GPIC_MFU_LOCAL_OPT=$FS GPIC_MFU_MUON_ADD=1 LADDER_WT=$WT LADDER_RUN_ROOT=$root LADDER_LAST_EVERY=$LAST BENCH_OUT=$root/bench_$(date +%s) \
  bash $WT/jobs/train_ladder.sh $size $d $nb $nt $S \
    --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr 0.00064 --model.optimizer.init_args.muon_lr 1.36e-2 \
    --model.optimizer.init_args.weight_decay 0 --model.optimizer.init_args.muon_weight_decay 0.002752 \
    --model.lr_scheduler.init_args.num_warmup_steps $((S / 100)) --model.lr_scheduler.init_args.num_cooldown_steps $(( (S / 5) / 32 * 32 )) \
    --model.lr_scheduler.init_args.max_steps $S --model.lr_scheduler.init_args.min_lr 3e-5 \
    --data.train_batch_size $MB --trainer.accumulate_grad_batches $((1024 / 8 / MB)) --data.train_dataset.init_args.shared_bucket true \
    --data.train_dataset.init_args.tar_pattern 'gpic_train_{00000..07999}.tar' \
    --trainer.logger.init_args.name smoke_saveterm_${tag} --trainer.logger.init_args.id smoke-saveterm-$tag-$(date +%m%d%H%M)
}
case_() {   # <tag> <cut after s> <d> <nb> <nt> <steps> <last every> <mb> <fsdp>
  local tag=$1 T=$2; shift 2
  echo "== $tag: launch 1 under timeout ${T}s ($(date +%T))"
  local t0=$(date +%s)
  ( export -f run; export OUT WT WC; timeout --kill-after=300 ${T}s bash -c "run $tag $*" ) > $OUT/$tag.1.log 2>&1
  local rc=$? t1=$(date +%s)
  echo "$tag launch 1: rc $rc, exit $((t1 - t0 - T)) s after the TERM; $(grep -a -h '\[save_on_term\]' $OUT/$tag.1.log | head -1)"
  echo "== $tag: launch 2 ($(date +%T))"
  ( export -f run; export OUT WT WC; bash -c "run $tag $*" ) > $OUT/$tag.2.log 2>&1
  echo "$tag launch 2: rc $?; $(grep -a -h 'resuming from' $OUT/$tag.2.log | head -1)"
  grep -a -h "checkpoint step" $OUT/$tag/exp_ladder_d$1/launches.txt | tail -2 | sed "s/^/$tag launches: /"
}
case_ d9 240 640 9 2 4096 512 128 0     # last every 512: not max_steps/4 (the ms hook's interval; two equal hooks stop Lightning)
case_ d16 330 1536 16 4 1024 128 128 1
source $WC/envs/pixgen/bin/activate
python - $OUT <<'PY'
import glob, json, math, os, sys
out = sys.argv[1]
for tag in ("d9", "d16"):
    bs = sorted(glob.glob(f"{out}/{tag}/bench_*"))
    if len(bs) < 2:
        print(f"{tag}: bench dirs {len(bs)}"); continue
    rows = [{r["step"]: r for r in json.load(open(f"{b}/steps_rank0.json"))["rows"]} for b in bs[:2]]
    a, b = rows
    cut = min(b) if b else None
    print(f"{tag}: launch 1 steps {min(a)}-{max(a)}, launch 2 steps {cut}-{max(b) if b else None}; steps redone {max(a) + 1 - cut if cut is not None else None}")
    if cut is not None:
        la = [a[s]["loss"] for s in range(max(cut - 40, min(a)), cut) if s in a]
        lb = [b[s]["loss"] for s in range(cut, cut + 40) if s in b]
        print(f"{tag}: mean loss 40 steps before the cut {sum(la) / len(la):.4f}, 40 after the resume {sum(lb) / len(lb):.4f}; non-finite after: {sum(1 for r in b.values() if not math.isfinite(r['loss']))}")
PY
find $OUT -name "*.ckpt" -delete
echo SMOKE_SAVETERM_DONE $OUT
