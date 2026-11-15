#!/bin/bash
# CPU test of jobs/heldout_loss_lib.sh (heldout_run's control flow) with a stub in place of tools/heldout_loss.py:
#   bash tools/test_heldout_lib.sh      (any node; seconds; no GPU, no torch)
# The stub keeps a checkpoint's state as marker files: done/<name> (has its lines), writing/<name> (young and still being
# written: list leaves it out unless --min-age 0), late/<name> (a writing checkpoint that is whole once a worker has run),
# bad/<name> (its task fails). Each case gives the checkpoints, the expected return value, the number of worker starts
# and the checkpoints that must be done at the end.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
FAILS=0
mkdir -p $T/wt/tools
cat > $T/wt/tools/heldout_loss.py <<'PY'
import argparse, json, os, sys
S = os.environ["STUB_STATE"]
ap = argparse.ArgumentParser(); ap.add_argument("cmd")
for k in ("--out", "--data", "--n-val", "--n-train", "--skipped", "--tasks", "--counter", "--gpu", "--threads", "--splits"):
    ap.add_argument(k)
ap.add_argument("--min-age", type=float, default=600); ap.add_argument("--ckpt", nargs="*", default=None)
a = ap.parse_args()
has = lambda kind, c: os.path.exists(os.path.join(S, kind, os.path.basename(c)))
with open(os.path.join(S, "calls.log"), "a") as f:
    f.write(a.cmd + (" min-age-0" if a.cmd == "list" and a.min_age == 0 else "") + (" ckpt=" + ",".join(map(os.path.basename, a.ckpt)) if a.ckpt is not None else "") + "\n")
if a.cmd == "list":
    cks = a.ckpt if a.ckpt is not None else [l.strip() for l in open(os.path.join(S, "all.txt"))]
    young = [c for c in cks if has("writing", c) and a.min_age > 0]
    for c in cks:
        if c not in young and not has("done", c):
            print(json.dumps({"ckpt": c}))
    if a.skipped:
        open(a.skipped, "w").write("".join(c + "\n" for c in young))
elif a.cmd == "work":
    rc = 0
    for name in os.listdir(os.path.join(S, "late")):   # time has passed: these are whole now
        try: os.remove(os.path.join(S, "writing", name))
        except FileNotFoundError: pass
    for l in open(a.tasks):
        c = json.loads(l)["ckpt"]
        if has("bad", c):
            print(f"[heldout] {os.path.basename(c)}: FAILED"); rc = 1
        elif not has("done", c):
            open(os.path.join(S, "done", os.path.basename(c)), "w").close()
            print(f"[heldout] {os.path.basename(c)}: ema: val 0.1 train 0.1")
    sys.exit(rc)
else:
    sys.exit(3)
PY
case_() {   # <name> <mode: given|default> <want rc> <want worker starts> <want done names> -- <ckpt>[:state] ...
  local name=$1 mode=$2 want_rc=$3 want_work=$4 want_done=$5; shift 6
  local S=$T/$name; mkdir -p $S/{done,writing,late,bad} $S/wc/tmp/heldout_loss
  : > $S/calls.log; : > $S/all.txt; : > $S/wc/tmp/heldout_loss/data_v1_val4_train3_00787.pt
  local cks=() c st
  for c in "$@"; do
    st=${c#*:}; [ "$st" != "$c" ] || st=""; c=/runs/exp/${c%%:*}
    cks+=("$c"); echo "$c" >> $S/all.txt
    case $st in
      done|writing|bad) : > $S/$st/$(basename $c) ;;
      late) : > $S/writing/$(basename $c); : > $S/late/$(basename $c) ;;
    esac
  done
  [ $mode = given ] || cks=()
  ( export STUB_STATE=$S WC=$S/wc WT=$T/wt FETCHED_JOB_ID=$name
    set -eo pipefail   # as in the callers: no step of the function may end the script before its own verdict
    source $HERE/jobs/heldout_loss_lib.sh
    rc=0; heldout_run $S/out.jsonl 4 3 2 "${cks[@]}" > $S/log.txt 2>&1 || rc=$?
    echo $rc > $S/rc )
  local rc=$(cat $S/rc 2>/dev/null || echo none) work=$(grep -c '^work' $S/calls.log) done_=$(ls $S/done | sort | paste -sd,)
  if [ "$rc" = "$want_rc" ] && [ "$work" = "$want_work" ] && [ "$done_" = "$want_done" ]; then echo "PASS $name"
  else echo "FAIL $name: rc $rc (want $want_rc), worker starts $work (want $want_work), done '$done_' (want '$want_done')"; sed 's/^/    /' $S/log.txt $S/calls.log; FAILS=$((FAILS + 1)); fi
}
#      name            mode    rc work done            --  checkpoints
case_  all_whole       given   0  2    a,b,c           --  a b c
case_  nothing_to_do   given   0  0    a,b             --  a:done b:done
case_  none_given_done default 0  0    a               --  a:done
case_  late_whole      given   0  4    a,b,c           --  a b c:late
case_  never_whole     given   1  2    a,b             --  a b c:writing
case_  only_writing    given   1  0    ""              --  c:writing
case_  worker_fails    given   1  2    a               --  a b:bad
case_  default_whole   default 0  2    a,b             --  a b
case_  default_late    default 0  4    a,b             --  a b:late
case_  default_never   default 1  2    a               --  a b:writing
# the second pass asks only for the checkpoints the first list left out, and its logs do not write over the first pass's
grep -q '^list ckpt=c$' $T/late_whole/calls.log && [ -e $T/late_whole/wc/tmp/heldout_loss/work/late_whole/gpu0.pass2.log ] \
  && [ -e $T/late_whole/wc/tmp/heldout_loss/work/late_whole/gpu0.log ] && echo "PASS second pass" || { echo "FAIL second pass"; cat $T/late_whole/calls.log; FAILS=$((FAILS + 1)); }
grep -q '^list min-age-0$' $T/default_never/calls.log && grep -q 'still missing a line: 1; worker failures: 0' $T/default_never/log.txt \
  && echo "PASS final count" || { echo "FAIL final count"; cat $T/default_never/log.txt; FAILS=$((FAILS + 1)); }
grep -q 'pass 1: 0 tasks' $T/nothing_to_do/log.txt && ! grep -q 'pass 2' $T/nothing_to_do/log.txt && echo "PASS no worker without a task" || { echo "FAIL no worker without a task"; FAILS=$((FAILS + 1)); }
[ $FAILS = 0 ] && echo "TEST_HELDOUT_LIB PASS" || echo "TEST_HELDOUT_LIB FAIL: $FAILS"
exit $FAILS
