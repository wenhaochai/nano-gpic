#!/bin/bash
#SBATCH -J gpic-smoke-heldout5
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:30:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# GPU smoke of held-out run5 (gpic-28 2026-11-09; jobs/heldout_loss_lib.sh, tools/heldout_loss.py list): heldout_run on real
# checkpoints through the production path, with the cases that run4 got wrong. About 6 min on 8 GPUs; every pass has 8 tasks.
#  A  16 checkpoints into a new output file: 6 real ones that have lines in tmp/heldout_loss/heldout_loss.jsonl, 2 young
#     whole copies of a small one (run4 left such a file out for 600 s), and 8 young files that are cut when the job
#     lists them and whole (a rename, as the async writer does) while pass 1 runs. Expected: pass 1 scores 8 and names 8
#     as still being written, pass 2 scores those 8, return 0, 32 lines; every loss equals the line that run4 wrote for
#     the same weights (relative difference under 1e-4; printed).
#  B  one young cut file that stays cut: no task, no worker, return 1 (run4: return 0).
#  C  A's checkpoints again: nothing to do, no worker, return 0.
# Result: SMOKE_HELDOUT5 PASS | FAIL <reasons>. The copies are removed at the end; the lines stay in the smoke's directory.
# usage: smoke_heldout_run5.sh [worktree under gpic-wt, default heldout_loss_run5]
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
WT=$WC/project/gpic-wt/${1:-heldout_loss_run5}
[ -e "$WT/.git" ] || { echo "no worktree $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
source $WC/project/gpic/jobs/env.sh
export PYTHONPATH=$WT/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export NCCL_NVLS_ENABLE=0
source $WT/jobs/heldout_loss_lib.sh
JID=${FETCHED_JOB_ID:-${SLURM_JOB_ID:-manual}}
OUT=$WC/runs/gpic-speed/mfu_probe/heldout5_$(date +%m%d_%H%M); mkdir -p $OUT/runs
MAIN=$WC/tmp/heldout_loss/heldout_loss.jsonl
echo "code $(basename $WT) $(git -C $WT rev-parse --short HEAD) on $(hostname); out $OUT"
FAIL=""
# the 6 newest checkpoints with both lines from the compiled one-pass tool, and the newest small one as the copies' source
python - $MAIN > $OUT/pick.txt <<'PY'
import json, os, sys, zipfile
recs = [json.loads(l) for l in open(sys.argv[1])]
ok = {}
for r in recs:
    if r.get("data") == "data_v1_val3072_train3072_00787" and r.get("n_val") == 3072 and r.get("n_train") == 3072 and r.get("compiled") and r.get("one_pass"):
        ok.setdefault(r["ckpt"], {})[r["weights"]] = r
cks = [c for c, w in ok.items() if set(w) == {"ema", "raw"} and os.path.isfile(c) and zipfile.is_zipfile(c)]
cks.reverse()   # newest first
src = next((c for c in cks if ok[c]["raw"]["d"] <= 768), None) or min(cks, key=os.path.getsize)
real = [c for c in cks if c != src][:6]
assert len(real) == 6, f"only {len(real)} scored checkpoints left on disk"
print("src", src)
for c in real:
    print("real", c)
PY
[ $? = 0 ] || { echo "SMOKE_HELDOUT5 FAIL no-checkpoints $OUT"; exit 1; }
cat $OUT/pick.txt
SRC=$(awk '$1 == "src" {print $2}' $OUT/pick.txt); mapfile -t REAL < <(awk '$1 == "real" {print $2}' $OUT/pick.txt)
SIZE=$(basename $(dirname $SRC)); SIZE=${SIZE#exp_ladder_}; NAME=$(basename $SRC)
mk() {   # <tag>: a run directory with the source run's config; prints the checkpoint's path
  local d=$OUT/runs/$1/exp_ladder_$SIZE; mkdir -p $d; cp $(dirname $SRC)/config-TrainerFn.FITTING-*.yaml $d/; echo $d/$NAME
}
WHOLE=$(mk young0); cp $SRC $WHOLE                      # a new file: young and whole
YOUNG=($WHOLE $(mk young1)); ln $WHOLE ${YOUNG[1]}
LATE=(); for i in 0 1 2 3 4 5 6 7; do f=$(mk late$i); head -c 50000000 $WHOLE > $f; LATE+=($f); done   # young and cut
CUT=$(mk cut); head -c 50000000 $WHOLE > $CUT
ls -l --time-style=+%T ${YOUNG[@]} ${LATE[0]} $CUT | awk '{print $2, $5, $6, $7}'

echo "== A ($(date +%T))"
WORK_A=$WC/tmp/heldout_loss/work/${JID}_a
rm -rf $WC/tmp/heldout_loss/work/${JID}_{a,b,c}   # a second run of this job: the first run's logs would start the writers too soon
( until [ -e $WORK_A/gpu0.log ]; do sleep 1; done   # pass 1 has listed and started its workers: the writers finish now
  for f in "${LATE[@]}"; do ln $WHOLE $f.tmp$$ && mv -f $f.tmp$$ $f; done; echo "the 8 cut files are whole ($(date +%T))" ) &
LPID=$!
t0=$(date +%s)
( export FETCHED_JOB_ID=${JID}_a; set -e; heldout_run $OUT/a.jsonl 3072 3072 8 "${REAL[@]}" "${YOUNG[@]}" "${LATE[@]}" ) > $OUT/a.log 2>&1; RA=$?
kill $LPID 2> /dev/null; wait $LPID 2> /dev/null
cat $OUT/a.log; echo "A: return $RA in $(( $(date +%s) - t0 )) s"
[ $RA = 0 ] || FAIL="$FAIL A-return-$RA"
grep -q "pass 1: 8 tasks .*; 8 checkpoints still being written" $OUT/a.log || FAIL="$FAIL A-pass1"
grep -q "pass 2: 8 tasks .*; 0 checkpoints still being written" $OUT/a.log || FAIL="$FAIL A-pass2"
grep -q "still missing a line: 0; worker failures: 0" $OUT/a.log || FAIL="$FAIL A-count"

echo "== B ($(date +%T))"
touch $CUT   # still young, whatever time A took
( export FETCHED_JOB_ID=${JID}_b; set -e; heldout_run $OUT/b.jsonl 3072 3072 8 $CUT ) > $OUT/b.log 2>&1; RB=$?
cat $OUT/b.log; echo "B: return $RB"
[ $RB = 1 ] || FAIL="$FAIL B-return-$RB"
grep -q "still missing a line: 1; worker failures: 0" $OUT/b.log || FAIL="$FAIL B-count"
[ -z "$(ls $WC/tmp/heldout_loss/work/${JID}_b/gpu*.log 2> /dev/null)" ] && [ ! -s $OUT/b.jsonl ] || FAIL="$FAIL B-worker-started"

echo "== C ($(date +%T))"
( export FETCHED_JOB_ID=${JID}_c; set -e; heldout_run $OUT/a.jsonl 3072 3072 8 "${REAL[@]}" "${YOUNG[@]}" "${LATE[@]}" ) > $OUT/c.log 2>&1; RC=$?
cat $OUT/c.log; echo "C: return $RC"
[ $RC = 0 ] || FAIL="$FAIL C-return-$RC"
grep -q "pass 1: 0 tasks .*; 0 checkpoints still being written" $OUT/c.log || FAIL="$FAIL C-pass1"
[ -z "$(ls $WC/tmp/heldout_loss/work/${JID}_c/gpu*.log 2> /dev/null)" ] || FAIL="$FAIL C-worker-started"

echo "== values ($(date +%T))"
python - $MAIN $OUT/a.jsonl $SRC "${REAL[@]}" -- "${YOUNG[@]}" "${LATE[@]}" <<'PY' || FAIL="$FAIL values"
import json, sys
main, out, src = sys.argv[1:4]
rest = sys.argv[4:]; real, copies = rest[:rest.index("--")], rest[rest.index("--") + 1:]
ref = {}
for l in open(main):
    r = json.loads(l)
    if r.get("data") == "data_v1_val3072_train3072_00787" and r.get("compiled") and r.get("one_pass"):
        ref[(r["ckpt"], r["weights"])] = r
got = {}
for l in open(out):
    r = json.loads(l)
    assert (r["ckpt"], r["weights"]) not in got, f"two lines for {r['ckpt']} {r['weights']}"
    got[(r["ckpt"], r["weights"])] = r
want = {(c, w) for c in real + copies for w in ("ema", "raw")}
assert set(got) == want, f"lines {len(got)}, wanted {len(want)}; missing {sorted(want - set(got))[:3]}, extra {sorted(set(got) - want)[:3]}"
worst, bits = 0.0, 0
for (c, w), r in got.items():
    base = ref[(c if c in real else src, w)]
    for k in ("loss_val", "loss_train"):
        d = abs(r[k] / base[k] - 1)
        worst = max(worst, d); bits += r[k] == base[k]
        assert d < 1e-4, f"{c} {w} {k}: {r[k]} against run4's {base[k]}"
    assert r.get("one_pass") and r.get("compiled"), (c, w)
vals = {(r["loss_val"], r["loss_train"]) for (c, w), r in got.items() if c in copies}
print(f"[smoke] {len(got)} lines; largest relative difference from run4's lines {worst:.2e}; equal to the last digit in {bits} of {2 * len(got)} values; "
      f"the 10 copies of one checkpoint give {len(vals)} distinct value pair(s)")
PY

rm -rf $OUT/runs
if [ -z "$FAIL" ]; then echo "SMOKE_HELDOUT5 PASS $OUT"; else echo "SMOKE_HELDOUT5 FAIL$FAIL $OUT"; exit 1; fi
