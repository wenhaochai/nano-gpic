# Evaluation chain (gpic-28 2026-11-09, run6; owner: 「一定要处理空闲」): sourced by jobs/evals_gmflow.sh and jobs/evals_group.sh.
# run5's smoke left all 8 GPUs idle ~58 s at every change from one predict to the next (end of one process, start of the next). Here predict
# k+1 starts when predict k samples (its batch 1), sets up and compiles on its own batch 0, and waits at its batch 1 until predict k's last
# batch is gathered (lightning_model_gpic.py _eval_gate, save_images.py GPIC_EVAL_DONE_FILE); predict k saves and exits beside it. A scorer
# loop scores the predicted tasks in order, one at a time, beside the predicts; at most CHAIN_MAX_UNSCORED (default 3) predicted sample
# sets wait for it in node memory. The caller defines:
#   chain_next          sets CHAIN_I to the next task to run, or -1 when none is left (it may wait, e.g. for a raw view, and record skips)
#   chain_cmd <phase> <i>   runs task i's phase (predict or score) in the foreground and returns its status (eval_gmflow.sh EVAL_PHASE)
# chain_run needs EVAL_SHM_ROOT (a directory the caller removes at exit); afterwards CHAIN_FAILED holds the failed task indices.
_chain_launch() {   # <k> <i>: task i as the chain's k-th predict, in the background; CHAIN_PID
  local k=$1 i=$2 G=$EVAL_SHM_ROOT/chain
  ( export GPIC_EVAL_STARTED_FILE=$G/started_$k GPIC_EVAL_DONE_FILE=$G/done_$k
    if [ $k -gt 0 ]; then export GPIC_EVAL_GO_FILE=$G/done_$((k - 1)); else unset GPIC_EVAL_GO_FILE; fi
    chain_cmd predict $i ) &
  CHAIN_PID=$!
}
chain_run() {
  local G=$EVAL_SHM_ROOT/chain k=0 i nxt pid npid rc SC f
  CHAIN_FAILED=""
  mkdir -p $G
  ( j=0   # the scorer: pred_<j> = "ok <i>" or "fail <i>", written in chain order; score_<j> = "<status> <i>"
    while :; do
      until [ -e $G/pred_$j ]; do [ -e $G/end ] && [ ! -e $G/pred_$j ] && exit 0; sleep 1; done
      read -r st ti < $G/pred_$j
      if [ "$st" = ok ]; then chain_cmd score $ti; echo "$? $ti" > $G/score_$j.tmp; else echo "skip $ti" > $G/score_$j.tmp; fi
      mv $G/score_$j.tmp $G/score_$j
      j=$((j + 1))
    done ) &
  SC=$!
  chain_next; i=$CHAIN_I
  if [ $i -ge 0 ]; then _chain_launch 0 $i; pid=$CHAIN_PID; fi
  while [ $i -ge 0 ]; do
    until [ -e $G/started_$k ] || ! kill -0 $pid 2>/dev/null; do sleep 1; done
    chain_next; nxt=$CHAIN_I; npid=""
    if [ $nxt -ge 0 ]; then
      while [ $(( $(ls $G | grep -c '^pred_[0-9]*$') - $(ls $G | grep -c '^score_[0-9]*$') )) -ge ${CHAIN_MAX_UNSCORED:-3} ]; do sleep 2; done
      _chain_launch $((k + 1)) $nxt; npid=$CHAIN_PID
    fi
    wait $pid; rc=$?
    touch $G/started_$k $G/done_$k   # a predict that stopped before its last batch still opens the next one's gate
    if [ $rc -eq 0 ]; then echo "ok $i" > $G/pred_$k.tmp; else echo "fail $i" > $G/pred_$k.tmp; CHAIN_FAILED+=" $i"; echo "predict of task $i failed"; fi
    mv $G/pred_$k.tmp $G/pred_$k
    k=$((k + 1)); i=$nxt; pid=$npid
  done
  touch $G/end
  wait $SC
  for f in $G/score_*; do
    [ -e "$f" ] || continue
    read -r rc i < $f
    [ "$rc" = 0 ] || [ "$rc" = skip ] || { CHAIN_FAILED+=" $i"; echo "scoring of task $i failed"; }
  done
}
