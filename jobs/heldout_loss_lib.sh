# Shared by jobs/eval_heldout_loss.sh and jobs/smoke_heldout_loss.sh (sourced, after env.sh and PYTHONPATH).
# heldout_run <out.jsonl> <n_val> <n_train> <gpus> [checkpoints...]: prep the data if missing (GPU 0, about 2 min),
# list the tasks (all ladders, or the given checkpoints), run one worker per GPU on a shared counter, then list again:
# returns 0 only if no task is left.
# run5 (gpic-28 2026-11-09, after gen's report on w2372, w2395 and w2423): the list takes a young checkpoint that is whole
# (tools/heldout_loss.py complete), so a job that starts right after its last training scores that run's final checkpoint;
# a list without a task starts no worker (w2423: 8 workers took the 8 GPUs to find nothing); a checkpoint that was still
# being written at the first list gets a second pass after the first, and without a line at the end it counts as missing
# (before: it was left out and the job returned 0).
_heldout_pass() {   # <pass> <tasks.jsonl>: one worker per GPU on a shared counter; returns 1 if a worker failed
  local pass=$1 tasks=$2 pids=() i rc=0 sfx=""
  [ $pass = 1 ] || sfx=.pass$pass
  rm -f $work/counter
  for ((i = 0; i < ng; i++)); do
    python $tool work --out $out --data $data --n-val $nv --n-train $nt --tasks $tasks --counter $work/counter \
      --gpu $i --threads 8 ${HELDOUT_SPLITS:+--splits $HELDOUT_SPLITS} > $work/gpu$i$sfx.log 2>&1 &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do wait ${pids[$i]} || { echo "[heldout] worker $i exit $? (log $work/gpu$i$sfx.log)"; rc=1; }; done
  for ((i = 0; i < ng; i++)); do grep -h "^\[heldout\].*: ema: val\|FAILED" $work/gpu$i$sfx.log || true; done
  return $rc
}
heldout_run() {
  local out=$1 nv=$2 nt=$3 ng=$4; shift 4
  local tool=$WT/tools/heldout_loss.py dir=$WC/tmp/heldout_loss
  local jid=${FETCHED_JOB_ID:-${SLURM_JOB_ID:-manual}}
  local work=$dir/work/$jid
  mkdir -p $work
  echo "[heldout] job $jid code $(git -C $WT rev-parse HEAD) $(git -C $WT status --porcelain --untracked-files=no | wc -l) modified tracked files; tool sha1 $(sha1sum < $tool | cut -c1-12)"
  local data=$dir/data_v1_val${nv}_train${nt}_00787.pt
  [ -f $data ] || python $tool prep --n-val $nv --n-train $nt --procs 32 --data $data
  local all=(); [ $# -eq 0 ] || all=(--ckpt "$@")
  local ck=("${all[@]}") rc=0 pass n sk=()
  : > $work/tasks.jsonl
  for pass in 1 2; do
    python $tool list --out $out --data $data --n-val $nv --n-train $nt --skipped $work/skipped.txt "${ck[@]}" > $work/tasks$pass.jsonl
    cat $work/tasks$pass.jsonl >> $work/tasks.jsonl
    n=$(wc -l < $work/tasks$pass.jsonl)
    mapfile -t sk < $work/skipped.txt
    echo "[heldout] pass $pass: $n tasks in $work/tasks$pass.jsonl; ${#sk[@]} checkpoints still being written"
    if [ $n -gt 0 ]; then _heldout_pass $pass $work/tasks$pass.jsonl || rc=1; fi
    if [ ${#sk[@]} -eq 0 ]; then break; fi
    ck=(--ckpt "${sk[@]}")   # pass 2: only these, after the time pass 1 took
  done
  printf '%s\n' "${sk[@]}" > $work/skipped.txt
  # what this job was to score and still lacks a line: its tasks, and the checkpoints it left out as still being written
  local left=$(python $tool list --out $out --data $data --n-val $nv --n-train $nt --min-age 0 "${all[@]}" 2>/dev/null \
    | python -c "
import json, sys
mine = {json.loads(l)['ckpt'] for l in open('$work/tasks.jsonl')} | {l.strip() for l in open('$work/skipped.txt') if l.strip()}
print(sum(json.loads(l)['ckpt'] in mine for l in sys.stdin))")
  echo "[heldout] tasks of this job still missing a line: $left; worker failures: $rc"
  [ "$left" = 0 ] && [ $rc = 0 ]
}
