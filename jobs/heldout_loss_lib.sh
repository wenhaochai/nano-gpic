# Shared by jobs/eval_heldout_loss.sh and jobs/smoke_heldout_loss.sh (sourced, after env.sh and PYTHONPATH).
# heldout_run <out.jsonl> <n_val> <n_train> <gpus> [checkpoints...]: prep the data if missing (GPU 0, about 2 min),
# list the tasks (all ladders, or the given checkpoints), run one worker per GPU on a shared counter, then list again:
# returns 0 only if no task is left.
heldout_run() {
  local out=$1 nv=$2 nt=$3 ng=$4; shift 4
  local tool=$WT/tools/heldout_loss.py dir=$WC/tmp/heldout_loss
  local jid=${FETCHED_JOB_ID:-${SLURM_JOB_ID:-manual}}
  local work=$dir/work/$jid
  mkdir -p $work
  echo "[heldout] job $jid code $(git -C $WT rev-parse HEAD) $(git -C $WT status --porcelain --untracked-files=no | wc -l) modified tracked files; tool sha1 $(sha1sum < $tool | cut -c1-12)"
  local data=$dir/data_v1_val${nv}_train${nt}_00787.pt
  [ -f $data ] || python $tool prep --n-val $nv --n-train $nt --procs 32 --data $data
  local ck=(); [ $# -eq 0 ] || ck=(--ckpt "$@")
  python $tool list --out $out --data $data --n-val $nv --n-train $nt "${ck[@]}" > $work/tasks.jsonl
  local n=$(wc -l < $work/tasks.jsonl)
  echo "[heldout] $n tasks in $work/tasks.jsonl"
  rm -f $work/counter
  local pids=() i
  for ((i = 0; i < ng; i++)); do
    python $tool work --out $out --data $data --n-val $nv --n-train $nt --tasks $work/tasks.jsonl --counter $work/counter \
      --gpu $i --threads 8 ${HELDOUT_SPLITS:+--splits $HELDOUT_SPLITS} > $work/gpu$i.log 2>&1 &
    pids+=($!)
  done
  local rc=0
  for i in "${!pids[@]}"; do wait ${pids[$i]} || { echo "[heldout] worker $i exit $? (log $work/gpu$i.log)"; rc=1; }; done
  grep -h "^\[heldout\].*: ema: val\|FAILED" $work/gpu*.log || true
  local left=$(python $tool list --out $out --data $data --n-val $nv --n-train $nt --min-age 0 "${ck[@]}" 2>/dev/null \
    | python -c "import json, sys; done = {json.loads(l)['ckpt'] for l in open('$work/tasks.jsonl')}; print(sum(json.loads(l)['ckpt'] in done for l in sys.stdin))")
  echo "[heldout] tasks of this job still missing a line: $left; worker failures: $rc"
  [ "$left" = 0 ] && [ $rc = 0 ]
}
