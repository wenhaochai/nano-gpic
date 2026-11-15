#!/bin/bash
#SBATCH -J gpic-smoke-predrepro
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:40:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The one check of jobs/smoke_ckpt_dedup.sh (w2430) that failed, done again with a criterion that fits it (gpic-28 2026-11-09).
# w2430 wanted equal samples (md5) from a width-1536 predict of a twins-layout checkpoint and of an old-layout copy. The two
# files hold the same 500 tensors bit for bit, and the samples differed (3584 of 4096 images). The width-1536 evaluation was
# already known to differ run to run (FD spread 0.005% over five runs), the width-640 one repeats bit for bit
# (logs/mfu_eval_idle.yaml), so that check could not tell the layout from the predict's own difference. Here, on the
# checkpoints w2430 kept, with eval-official-run11, 16 batches per GPU (4096 samples), each predict in its own run root:
#  d9   width 640, step 4096: the old-layout copy, the twins file, the old-layout copy again. The predict repeats, so the
#       three sample files must be equal bit for bit: the test of the layout.
#  d16  width 1536, step 1024: the same three. Where the two old-layout predicts differ, the twins file must differ from
#       them no more than they differ from each other (tools/mfu/check_pred_repro.py, in onelast_cap_run).
#  d16:hashseed, d16:deterministic: the twins file and the copy under PYTHONHASHSEED=0, and under Lightning's
#       deterministic=warn (deterministic algorithms, cuBLAS workspace config): reported only, to find what differs.
# About 20 min. Result: SMOKE_PRED_REPRO PASS | FAIL <reasons>. On PASS the checkpoints w2430 kept are deleted.
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
G=$WC/project/gpic-wt
EWT=$G/eval_official_run11 TWT=$G/onelast_cap_run
SRC=$WC/runs/gpic-speed/mfu_probe/ckptdedup_1109_1652
for w in $EWT $TWT; do
  [ -z "$(git -C $w status --porcelain --untracked-files=no)" ] || { echo "$w has local changes"; exit 2; }
  echo "code $(basename $w) $(git -C $w rev-parse --short HEAD)"
done
T16=$SRC/d16/exp_ladder_d1536/ms-step=001024.ckpt O16=$SRC/old_view/d16/exp_ladder_d1536/ms-step=001024.ckpt
T9=$SRC/d9/exp_ladder_d640/ms-step=004096.ckpt
for f in $T16 $O16 $T9; do [ -f $f ] && [ ! -L $f ] || { echo "no checkpoint $f"; exit 2; }; done
source $WC/project/gpic/jobs/env.sh
OUT=$WC/runs/gpic-speed/mfu_probe/predrepro_$(date +%m%d_%H%M); mkdir -p $OUT
MAN=$OUT/manifest.txt; : > $MAN
FAIL=""
SHM=$(mktemp -d /dev/shm/gpic_predrepro_XXXXXX)
_prev_exit_cmd=$(trap -p EXIT | sed -E "s/^trap -- '(.*)' EXIT$/\1/; s/'\\\\''/'/g")
_on_exit() { rm -rf -- "$SHM"; eval "$_prev_exit_cmd"; }
trap _on_exit EXIT

O9=$OUT/old_d9/ms-step=004096.ckpt
python - $T9 $O9 $T16 $O16 <<'PY' || FAIL="$FAIL old-layout-copy"
import os, sys, torch
t9, o9, t16, o16 = sys.argv[1:5]
os.makedirs(os.path.dirname(o9), exist_ok=True)
ck = torch.load(t9, map_location="cpu", mmap=True, weights_only=False)
ck["state_dict"] = {k: v.clone() for k, v in ck["state_dict"].items()}   # every tensor on its own storage: the old layout
torch.save(ck, o9)
ok = True
for name, t, o in (("d9", t9, o9), ("d16", t16, o16)):
    a, b = os.path.getsize(o), os.path.getsize(t)
    print(f"{name}: old-layout copy {a} bytes, twins file {b} bytes (ratio {b / a:.4f})")
    ok &= a > 1.9 * b
sys.exit(0 if ok else 1)
PY

pred() {   # <group> <name> <checkpoint> <size> <d> <nb> <nt> <step> [VAR=value]...: one predict, 16 batches per GPU, samples in $SHM
  local g=$1 name=$2 ck=$3 size=$4 d=$5 nb=$6 nt=$7 step=$8; shift 8
  local tag=${g//:/-}_$name
  local root=$OUT/p/$tag run=$OUT/p/$tag/exp_ladder_$size
  mkdir -p $run && ln -s $ck $run/ms-step=$(printf %06d $step).ckpt
  local before=$(ls -d $SHM/task_* 2>/dev/null | sort) t0=$(date +%s)
  env EVAL_LIMIT_BATCHES=16 "$@" EVAL_PHASE=predict EVAL_SHM_ROOT=$SHM GMFLOW_WT=$EWT LADDER_RUN_ROOT=$root EVAL_SIZE=$size EVAL_D=$d EVAL_NB=$nb EVAL_NT=$nt \
    EVAL_SPREAD=image EVAL_NOISE=official bash $EWT/jobs/eval_gmflow.sh 0 $step euler 50 > $OUT/pred_$tag.log 2>&1
  local rc=$?
  local new=$(comm -13 <(echo "$before") <(ls -d $SHM/task_* 2>/dev/null | sort) | head -1)   # this predict's own directory
  local npy=""; [ -z "$new" ] || npy=$(find $new -name "output.np[yz]" | head -1)
  [ -n "$npy" ] || npy=$(find $run -name "output.np[yz]" | head -1)   # node memory short: the samples are beside the run
  local o=$(ls -d $run/eval_* 2>/dev/null | head -1)
  echo "predict $g $name: rc $rc in $(( $(date +%s) - t0 )) s; $(head -1 $o/sample_check.txt 2>/dev/null)"
  if [ $rc = 0 ] && [ -n "$npy" ]; then echo "$g $name $npy" >> $MAN
  elif [ "${g/:/}" = "$g" ]; then FAIL="$FAIL predict-$tag"; tail -5 $OUT/pred_$tag.log
  else echo "predict $g $name failed: this setting is not compared"; tail -3 $OUT/pred_$tag.log; fi
}
echo "== d9, step 4096 ($(date +%T))"
pred d9 old  $O9 d640 640 9 2 4096
pred d9 link $T9 d640 640 9 2 4096
pred d9 old2 $O9 d640 640 9 2 4096
echo "== d16, step 1024 ($(date +%T))"
pred d16 old  $O16 d1536 1536 16 4 1024
pred d16 link $T16 d1536 1536 16 4 1024
pred d16 old2 $O16 d1536 1536 16 4 1024
echo "== d16 with PYTHONHASHSEED=0 ($(date +%T))"
pred d16:hashseed link $T16 d1536 1536 16 4 1024 PYTHONHASHSEED=0
pred d16:hashseed old  $O16 d1536 1536 16 4 1024 PYTHONHASHSEED=0
echo "== d16 with deterministic=warn ($(date +%T))"
pred d16:deterministic link $T16 d1536 1536 16 4 1024 "EVAL_LIMIT_BATCHES=16 --trainer.deterministic warn"
pred d16:deterministic old  $O16 d1536 1536 16 4 1024 "EVAL_LIMIT_BATCHES=16 --trainer.deterministic warn"
ND=$(cat $OUT/p/d16-deterministic_*/exp_ladder_d1536/eval_*/predict.log 2>/dev/null | grep -a -o "[A-Za-z_0-9.]* does not have a deterministic implementation" | sort | uniq -c | tr -s ' ' | tr '\n' ';')
echo "deterministic=warn: operations without a deterministic implementation in the predict logs:${ND:- none}"

echo "== compare ($(date +%T))"
OMP_NUM_THREADS=16 python $TWT/tools/mfu/check_pred_repro.py $MAN 2>&1 | tee $OUT/check.log
[ ${PIPESTATUS[0]} = 0 ] || FAIL="$FAIL check"

if [ -z "$FAIL" ]; then
  find $SRC $OUT -name "*.ckpt" -delete
  echo "SMOKE_PRED_REPRO PASS $OUT"
else
  mkdir -p $OUT/samples
  while read -r g n f; do cp $f $OUT/samples/${g//:/-}_$n.npy; done < $MAN
  echo "SMOKE_PRED_REPRO FAIL$FAIL $OUT (samples kept in $OUT/samples, the checkpoints kept in $SRC)"
  exit 1
fi
