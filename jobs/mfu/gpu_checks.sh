# Sourced by the MFU jobs: fail-fast checks of the custom GPU code before the arms, side by side on two GPUs (about 1 min):
# the FA3 ops (tools/mfu/fa3_check.py) and the compiled resize kernel on real images (tools/mfu/tests/test_pil_exact_gpu.py).
# usage: gpu_checks <worktree> <out dir>   (needs the array G of the job's GPUs); returns 1 when a check fails
gpu_checks() {
  local wt=$1 out=$2
  ( cd $wt && CUDA_VISIBLE_DEVICES=${G[0]} timeout 300 python tools/mfu/fa3_check.py > $out/fa3_check.log 2>&1 ) &
  local p1=$!
  ( cd $wt && CUDA_VISIBLE_DEVICES=${G[1]} timeout 300 python tools/mfu/tests/test_pil_exact_gpu.py 64 32 > $out/resize_check.log 2>&1 ) &
  local p2=$!
  wait $p1; local r1=$?
  wait $p2; local r2=$?
  tail -2 $out/fa3_check.log $out/resize_check.log
  [ $r1 -eq 0 ] && [ $r2 -eq 0 ]
}
