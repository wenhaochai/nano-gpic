# Sourced by the MFU probe jobs: run single-GPU tasks from a list on all of the job's GPUs, so that no GPU waits for another
# (controller, 2026-11-06: a wq job keeps every GPU it asks for busy). One worker per GPU takes the next line of the task
# file when it is free; list the longest tasks first. Each line is a bash command; it sees CUDA_VISIBLE_DEVICES = one GPU.
# usage: pool <task file> <log dir>   (needs the array G of the job's GPUs)
pool() {
  local tasks=$1 logs=$2 n next
  n=$(grep -c . "$tasks")
  next=$logs/.pool_next
  echo 0 > $next
  for gpu in "${G[@]}"; do
    (
      while true; do
        i=$(flock $next.lock bash -c "i=\$(cat $next); echo \$((i + 1)) > $next; echo \$i")
        [ "$i" -ge "$n" ] && break
        line=$(grep . "$tasks" | sed -n "$((i + 1))p")
        echo "[$(date +%T)] gpu $gpu start task $i: $line" >> $logs/pool.log
        CUDA_VISIBLE_DEVICES=$gpu bash -c "$line"
        echo "[$(date +%T)] gpu $gpu end task $i rc=$?" >> $logs/pool.log
      done
    ) &
  done
  wait
}
