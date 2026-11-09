#!/bin/bash
#SBATCH -J gpic-smoke-ckptdedup
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:55:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Smoke of the twins layout before it goes to production (gpic-28 2026-11-09, infra scan: 4.85 TiB of checkpoints in 48 h):
# a no-EMA run's EMA slot holds the raw weights, and ckpt-dedup-cap / ckpt-dedup-sched write that copy once. The three
# readers of the files run on what the two training lines write, about 40 min in all:
#  1 training as the sweeps run it, cut by TERM and resumed (jobs/smoke_save_on_term.sh's cases): d9 on ckpt-dedup-cap
#    (DDP, 4096 steps, last.ckpt every 512) and d16 on ckpt-dedup-sched (FSDP2, 1024 steps, every 128)
#  2 tools/mfu/check_ckpt_dedup.py: every file's layout and size, the final ms file against last.ckpt, the resume, and
#    the stall of the steps that hold a scheduled save against the old layout's smoke w2414
#  3 eval-official-run11: tools/make_raw_view.py must link the d16 checkpoint (no second copy), and the predict's samples
#    from the link must equal the samples from an old-layout copy of the same weights (16 batches per GPU, md5)
#  4 heldout-loss-run4: one pass (two records from the raw weights) against two passes on all 8 ms files: losses, time
# Result: SMOKE_CKPT_DEDUP PASS | FAIL <reasons>; the checkpoints are deleted on PASS and kept on FAIL.
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
G=$WC/project/gpic-wt
CWT=$G/ckptdedup_cap_run SWT=$G/ckptdedup_sched_run EWT=$G/eval_official_run11 HWT=$G/heldout_loss_run4
for w in $CWT $SWT $EWT $HWT; do
  [ -z "$(git -C $w status --porcelain --untracked-files=no)" ] || { echo "$w has local changes"; exit 2; }
  echo "code $(basename $w) $(git -C $w rev-parse --short HEAD)"
done
OUT=$WC/runs/gpic-speed/mfu_probe/ckptdedup_$(date +%m%d_%H%M); mkdir -p $OUT
FAIL=""
run() {   # <worktree> <tag> <d> <nb> <nt> <steps> <last every> <mb> <fsdp>: one launch of the sweep path
  local WT=$1 tag=$2 d=$3 nb=$4 nt=$5 S=$6 LAST=$7 MB=$8 FS=$9 root=$OUT/$2 size=d$3
  mkdir -p $root/exp_ladder_$size; [ -s $root/exp_ladder_$size/shard_seed.txt ] || echo 20261106 > $root/exp_ladder_$size/shard_seed.txt
  GPIC_MFU_NO_EMA=1 LD_PRELOAD=$WC/envs/nccl/2.27.7/libnccl.so.2 \
  GPIC_MFU_FSDP=$FS GPIC_MFU_FSDP_GROUP=1 GPIC_MFU_FSDP_NOSYNC=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_MFU_PATCHIFY=$FS \
  GPIC_MFU_NS_FUSED=1 GPIC_MFU_LOCAL_OPT=$FS GPIC_MFU_MUON_ADD=1 LADDER_WT=$WT LADDER_RUN_ROOT=$root LADDER_LAST_EVERY=$LAST BENCH_OUT=$root/bench_$(date +%s) \
  exec bash $WT/jobs/train_ladder.sh $size $d $nb $nt $S \
    --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr 0.00064 --model.optimizer.init_args.muon_lr 1.36e-2 \
    --model.optimizer.init_args.weight_decay 0 --model.optimizer.init_args.muon_weight_decay 0.002752 \
    --model.lr_scheduler.init_args.num_warmup_steps $((S / 100)) --model.lr_scheduler.init_args.num_cooldown_steps $(( (S / 5) / 32 * 32 )) \
    --model.lr_scheduler.init_args.max_steps $S --model.lr_scheduler.init_args.min_lr 3e-5 \
    --data.train_batch_size $MB --trainer.accumulate_grad_batches $((1024 / 8 / MB)) --data.train_dataset.init_args.shared_bucket true \
    --data.train_dataset.init_args.tar_pattern 'gpic_train_{00000..07999}.tar' \
    --trainer.logger.init_args.name smoke_ckptdedup_${tag} --trainer.logger.init_args.id smoke-ckptdedup-$tag-$(date +%m%d%H%M)
}
case_() {   # <worktree> <tag> <cut after s> <d> <nb> <nt> <steps> <last every> <mb> <fsdp>
  local wt=$1 tag=$2 T=$3; shift 3
  echo "== $tag on $(basename $wt): launch 1 under timeout ${T}s ($(date +%T))"
  local t0=$(date +%s)
  ( export -f run; export OUT WC; timeout --kill-after=300 ${T}s bash -c "run $wt $tag $*" ) > $OUT/$tag.1.log 2>&1
  local rc=$? t1=$(date +%s)
  echo "$tag launch 1: rc $rc, exit $((t1 - t0 - T)) s after the TERM; $(grep -a -h '\[save_on_term\]' $OUT/$tag.1.log | head -1)"
  [ $rc = 124 ] || FAIL="$FAIL $tag-launch1-rc$rc"
  echo "== $tag: launch 2 ($(date +%T))"
  ( export -f run; export OUT WC; bash -c "run $wt $tag $*" ) > $OUT/$tag.2.log 2>&1
  rc=$?
  echo "$tag launch 2: rc $rc; $(grep -a -h 'resuming from' $OUT/$tag.2.log | head -1)"
  [ $rc = 0 ] || FAIL="$FAIL $tag-launch2-rc$rc"
  grep -a -h "checkpoint step" $OUT/$tag/exp_ladder_d$1/launches.txt | tail -2 | sed "s/^/$tag launches: /"
}
# the training sources env.sh itself (it owns and removes the job's TMPDIR, as in a sweep job); this shell sources it after
case_ $CWT d9 240 640 9 2 4096 512 128 0     # last every 512 and 128: not max_steps/4 (the ms hook's interval; two equal hooks stop Lightning)
case_ $SWT d16 330 1536 16 4 1024 128 128 1

source $WC/project/gpic/jobs/env.sh
SHM=$(mktemp -d /dev/shm/gpic_dedup_XXXXXX)
_prev_exit_cmd=$(trap -p EXIT | sed -E "s/^trap -- '(.*)' EXIT$/\1/; s/'\\\\''/'/g")
_on_exit() { rm -rf -- "$SHM"; eval "$_prev_exit_cmd"; }
trap _on_exit EXIT
M=$OUT/d16/exp_ladder_d1536 N=$OUT/d9/exp_ladder_d640

echo "== 2 files, resume, save stalls: in the background beside the predicts ($(date +%T))"
OMP_NUM_THREADS=8 python $CWT/tools/mfu/check_ckpt_dedup.py $OUT $WC/runs/gpic-speed/mfu_probe/saveterm_1109_1150 > $OUT/check.log 2>&1 &
CPID=$!

echo "== 3 predict from the linked view and from an old-layout copy: d16 step 1024, 16 batches per GPU ($(date +%T))"
RAW_ROOT=$OUT RAW_SRC=d16 $WC/envs/pixgen/bin/python -I $EWT/tools/make_raw_view.py d1536 1024 2>&1 | tee $OUT/view.log
V=$OUT/raw_view/d16/exp_ladder_d1536/ms-step=001024.ckpt
if [ -L $V ] && [ "$(readlink -f $V)" = "$(readlink -f $M/ms-step=001024.ckpt)" ]; then echo "view: a link to the checkpoint"; else FAIL="$FAIL view-not-a-link"; fi
O=$OUT/old_view/d16/exp_ladder_d1536
python - $M $O 1024 > $OUT/old_view.log 2>&1 <<'PY' &
import glob, os, shutil, sys, torch
src, dst, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(dst, exist_ok=True)
name = f"ms-step={step:06d}.ckpt"
ck = torch.load(f"{src}/{name}", map_location="cpu", mmap=True, weights_only=False)
ck["state_dict"] = {k: v.clone() for k, v in ck["state_dict"].items()}   # every tensor on its own storage: the old layout
torch.save(ck, f"{dst}/{name}")
for c in glob.glob(f"{src}/config-*.yaml"):
    shutil.copy(c, dst)
a, b = os.path.getsize(f"{dst}/{name}"), os.path.getsize(f"{src}/{name}")
print(f"old-layout copy {a} bytes, twins file {b} bytes (ratio {b / a:.4f})")
sys.exit(0 if a > 1.9 * b else 1)
PY
OPID=$!
declare -A MD NPY
pred() {   # <name> <run root>: predict only (EVAL_PHASE=predict), samples in $SHM; records the samples' md5
  local name=$1 before=$(ls -d $SHM/task_* 2>/dev/null | sort)
  EVAL_PHASE=predict EVAL_SHM_ROOT=$SHM GMFLOW_WT=$EWT LADDER_RUN_ROOT=$2 EVAL_SIZE=d1536 EVAL_D=1536 EVAL_NB=16 EVAL_NT=4 \
    EVAL_SPREAD=image EVAL_NOISE=official EVAL_LIMIT_BATCHES=16 bash $EWT/jobs/eval_gmflow.sh 0 1024 euler 50 > $OUT/pred_$name.log 2>&1
  local rc=$?
  local new=$(comm -13 <(echo "$before") <(ls -d $SHM/task_* 2>/dev/null | sort) | head -1)   # this predict's own directory
  local npy=""; [ -z "$new" ] || npy=$(find $new -name "output.np[yz]" | head -1)
  [ -z "$npy" ] || { MD[$name]=$(md5sum < $npy | cut -c1-32); NPY[$name]=$npy; }
  local o=$(ls -d $2/exp_ladder_d1536/eval_* 2>/dev/null | head -1)
  echo "predict $name: rc $rc, md5 ${MD[$name]:-none}; $(cat $o/sample_check.txt 2>/dev/null | head -1); $(cat $o/phase_times.json 2>/dev/null)"
  [ -n "${MD[$name]:-}" ] || FAIL="$FAIL predict-$name-no-samples"
}
pred link $OUT/raw_view/d16
wait $OPID || FAIL="$FAIL old-view-copy"
cat $OUT/old_view.log
pred old $OUT/old_view/d16
if [ -n "${MD[link]:-}" ] && [ "${MD[link]:-}" = "${MD[old]:-}" ]; then echo "predict: the samples are equal (md5 ${MD[link]})"
else
  FAIL="$FAIL predict-md5"
  [ -z "${NPY[link]:-}" ] || [ -z "${NPY[old]:-}" ] || python - "${NPY[link]}" "${NPY[old]}" <<'PY'
import sys, numpy as np
a, b = (np.load(f) for f in sys.argv[1:3])
if a.shape != b.shape:
    print(f"predict: shapes differ {a.shape} {b.shape}")
else:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    print(f"predict: samples DIFFER: {a.shape}, share of different values {np.mean(d > 0):.6f}, largest difference {d.max()}, images with a difference {int((d.reshape(len(d), -1).max(1) > 0).sum())}")
PY
fi
rm -rf $SHM/task_*
wait $CPID || FAIL="$FAIL check"
cat $OUT/check.log

echo "== 4 held-out loss: one pass (GPUs 0-2) beside two passes (GPUs 3-7) ($(date +%T))"
D=$WC/tmp/heldout_loss/data_v1_val3072_train3072_00787.pt
HP=()
ho() {   # <out file> <gpu> <GPIC_HELDOUT_ONE_PASS> <checkpoints...>
  local o=$1 g=$2 one=$3; shift 3
  GPIC_HELDOUT_ONE_PASS=$one PYTHONPATH=$HWT/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    python $HWT/tools/heldout_loss.py work --out $o --data $D --n-val 3072 --n-train 3072 --gpu $g --threads 8 --ckpt "$@" > $OUT/heldout_gpu$g.log 2>&1 &
  HP+=($!)
}
N4=($N/ms-step=001024.ckpt $N/ms-step=002048.ckpt $N/ms-step=003072.ckpt $N/ms-step=004096.ckpt)
ho $OUT/heldout_one.jsonl 0 1 $M/ms-step=000256.ckpt $M/ms-step=000512.ckpt
ho $OUT/heldout_one.jsonl 1 1 $M/ms-step=000768.ckpt $M/ms-step=001024.ckpt
ho $OUT/heldout_one.jsonl 2 1 "${N4[@]}"
for i in 1 2 3 4; do ho $OUT/heldout_two.jsonl $((2 + i)) 0 $M/ms-step=$(printf %06d $((256 * i))).ckpt; done
ho $OUT/heldout_two.jsonl 7 0 "${N4[@]}"
for p in "${HP[@]}"; do wait $p || FAIL="$FAIL heldout-worker"; done
grep -h "FAILED\|worker [0-9]*:" $OUT/heldout_gpu*.log
python - $OUT <<'PY' || FAIL="$FAIL heldout-compare"
import collections, json, sys
o = sys.argv[1]
rd = lambda f: {(r["ckpt"], r["weights"]): r for r in map(json.loads, open(f"{o}/{f}"))}
one, two = rd("heldout_one.jsonl"), rd("heldout_two.jsonl")
ok = len(one) == len(two) == 16 and one.keys() == two.keys()
worst, t1, t2 = 0.0, collections.Counter(), collections.Counter()
for k in sorted(one.keys() & two.keys()):
    a, b = one[k], two[k]
    rel = max(abs(a[s] / b[s] - 1) for s in ("loss_val", "loss_train"))
    worst = max(worst, rel)
    ok &= a["one_pass"] is True and b["one_pass"] is False and a["compiled"] == b["compiled"]
    run = k[0].split("/")[-3]
    if k[1] == "raw":   # elapsed_s is the checkpoint's, on both of its records
        t1[run] += a["elapsed_s"]; t2[run] += b["elapsed_s"]
        e = two[(k[0], "ema")]
        print(f"held-out {run} {k[0].split('/')[-1]}: val {a['loss_val']:.6f} one pass, {b['loss_val']:.6f} two passes (raw), {e['loss_val']:.6f} (ema); "
              f"{a['elapsed_s']:.0f} s against {b['elapsed_s']:.0f} s")
ok &= worst <= 1e-5
print(f"held-out: {len(one)} and {len(two)} records; largest relative difference {worst:.2e}{' (exactly 0)' if worst == 0 else ''}; "
      + "; ".join(f"{r}: {t1[r]:.0f} s one pass, {t2[r]:.0f} s two passes ({100 * (t1[r] / t2[r] - 1):+.0f}%)" for r in sorted(t1)))
sys.exit(0 if ok else 1)
PY

if [ -z "$FAIL" ]; then
  find $OUT -name "*.ckpt" -delete
  echo "SMOKE_CKPT_DEDUP PASS $OUT"
else
  echo "SMOKE_CKPT_DEDUP FAIL$FAIL $OUT (checkpoints kept: $(du -sh $OUT | cut -f1))"
  exit 1
fi
