#!/bin/bash
#SBATCH -J gpic-prof-base
#SBATCH --gres=gpu:8
#SBATCH --time=02:30:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#FETCHABLE
# Detailed baseline profile of the official GPIC training step and the eval phases (who-ate-my-flops base).
#  P0 probes (GPU counters, clocks, topology)       P1 unprofiled x2 (cold + warm compile cache, nvidia-smi dmon)
#  P2 torch.profiler 8 ranks, steps 100-103          P3 nsys 8 ranks, same window (op NVTX with shapes, python sampling)
#  P4 single-GPU side by side: Inductor per-kernel bandwidth, compile debug + graph breaks, GEMM/SDPA/optimizer
#     microbenchmarks at the trace's shapes, correctness x3 + kernel swap + eager, eval phases (sampling, DINOv2)
#  P5 NCCL all-reduce bandwidth at the gradient sizes  P6 ncu per-kernel counters, sharded on 8 GPUs (if permitted)
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=/scratch/gpfs/GROUP/USER/project/gpic-wt/perf
WS=$WT/workspace-who-ate-my-flops
SHA=$(git -C $WT rev-parse --short HEAD)
R=$WS/runs/prof_$SHA
mkdir -p $R
log() { echo "[$(date +%T)] $*" | tee -a $R/phases.log; }
phase() { local name=$1; shift; local t0=$(date +%s); log "START $name"; "$@"; local rc=$?; log "END $name rc=$rc $(( $(date +%s) - t0 ))s"; }
export TORCHINDUCTOR_CACHE_DIR=$R/cache/inductor TRITON_CACHE_DIR=$R/cache/triton   # P1 = cold first run; kept: kernel sources

probe() {
  {
    echo "== nvidia params"; grep -i -E "profil|Restrict" /proc/driver/nvidia/params
    echo "== perf_event_paranoid $(cat /proc/sys/kernel/perf_event_paranoid) ptrace_scope $(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null)"
    nvidia-smi --query-gpu=index,name,driver_version,clocks.max.sm,clocks.sm,clocks.mem,power.limit,memory.total --format=csv
    nvidia-smi topo -m
    lscpu | grep -E "Model name|^CPU\(s\)|NUMA node"
    free -g | head -2
    module load nsight-systems/2025.3.1 && nsys --version
    NCU=$(ls /usr/local/cuda-13*/bin/ncu /usr/local/cuda-12.9/bin/ncu 2>/dev/null | sort -V | tail -1); echo "NCU=$NCU"; $NCU --version | tail -1
    CUDA_VISIBLE_DEVICES=0 timeout 180 $NCU --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed -c 1 \
      python -c "import torch;x=torch.ones(4096,4096,device='cuda');print((x@x).sum().item())" 2>&1 | tail -15
  } > $R/probe.txt 2>&1
}
phase p0_probe probe
NCU=$(grep -oP "^NCU=\K.*" $R/probe.txt)
NCU_OK=0; grep -q "sm__throughput" $R/probe.txt && ! grep -q "ERR_NVGPUCTRPERM" $R/probe.txt && NCU_OK=1
log "ncu counters available: $NCU_OK"
(cd $WT/baselines/PixelGen && PYTHONPATH=$PWD python -c "import sys; sys.path.insert(0, '$WT/tools'); from correctness import fixed_batch; fixed_batch()") \
  > $R/fixed_batch.log 2>&1

nvidia-smi dmon -s pucvmet -d 1 -o T > $R/p1_dmon.txt 2>&1 &
DMON=$!
phase p1_unprofiled_cold bash $WT/jobs/bench_train.sh $R/p1_unprofiled_cold
phase p1_unprofiled_warm bash $WT/jobs/bench_train.sh $R/p1_unprofiled_warm
kill $DMON 2>/dev/null
phase p2_torchprof bash $WT/jobs/bench_train.sh $R/p2_torchprof 100:103
module load nsight-systems/2025.3.1
GM=""; [ $NCU_OK = 1 ] && GM="--gpu-metrics-devices=all --gpu-metrics-frequency=20000"
mkdir -p $R/p3_nsys
phase p3_nsys env BENCH_CUDA_PROFILER=100:103 BENCH_WRAP="nsys profile -o $R/p3_nsys/report --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas,cudnn --pytorch=autograd-shapes-nvtx --python-sampling=true --python-sampling-frequency=1000 \
  --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --sample=process-tree \
  --cpuctxsw=process-tree $GM" bash $WT/jobs/bench_train.sh $R/p3_nsys

p4_inductor() { env CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29610 BENCH_DEVICES=1 BENCH_STEPS=20 TORCHINDUCTOR_PROFILE=1 \
  TORCHINDUCTOR_PROFILE_OUTPUT=$R/p4_inductor_bandwidth.txt TORCHINDUCTOR_CACHE_DIR=$R/cache/ind_prof \
  TRITON_CACHE_DIR=$R/cache/tri_prof bash $WT/jobs/bench_train.sh $R/p4_inductor; }
p4_compiledebug() { env CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29611 BENCH_DEVICES=1 BENCH_STEPS=20 \
  TORCH_LOGS="graph_breaks,recompiles,ddp_graphs,perf_hints" TORCH_COMPILE_DEBUG=1 TORCH_COMPILE_DEBUG_DIR=$R/p4_compiledebug/debug \
  TORCHINDUCTOR_CACHE_DIR=$R/cache/ind_dbg TRITON_CACHE_DIR=$R/cache/tri_dbg bash $WT/jobs/bench_train.sh $R/p4_compiledebug; }
p4_eval() { CUDA_VISIBLE_DEVICES=2 python $WT/tools/prof/eval_phase_prof.py $R/p4_evalphase > $R/p4_evalphase.log 2>&1; }
p4_micro() { CUDA_VISIBLE_DEVICES=3 python $WT/tools/prof/microbench.py $R/p2_torchprof/trace/rank0.json $R/p4_microbench.json \
  > $R/p4_microbench.log 2>&1; }
p4_corr() {
  cd $WT/baselines/PixelGen; export PYTHONPATH=$PWD; mkdir -p $R/corr
  for i in 1 2 3; do CUDA_VISIBLE_DEVICES=$((3 + i)) python $WT/tools/correctness.py record $R/corr/base$i.pt > $R/corr/base$i.log 2>&1 & done
  CUDA_VISIBLE_DEVICES=7 CORR_SDPA=math python $WT/tools/correctness.py record $R/corr/sdpa_math.pt > $R/corr/sdpa_math.log 2>&1 &
  wait
  CUDA_VISIBLE_DEVICES=4 CORR_COMPILE=0 python $WT/tools/correctness.py record $R/corr/eager.pt > $R/corr/eager.log 2>&1
  python $WT/tools/correctness.py derive $R/corr/tol.json $R/corr/base1.pt $R/corr/base2.pt $R/corr/base3.pt > $R/corr/derive_3runs.log 2>&1
  python $WT/tools/correctness.py derive $R/corr/tol_kernelswap.json $R/corr/base1.pt $R/corr/base2.pt $R/corr/base3.pt \
    $R/corr/sdpa_math.pt > $R/corr/derive_kernelswap.log 2>&1
}
phase p4_inductor p4_inductor & phase p4_compiledebug p4_compiledebug & phase p4_eval p4_eval &
phase p4_micro p4_micro & phase p4_corr p4_corr & wait

phase p5_nccl torchrun --standalone --nproc_per_node 8 $WT/tools/prof/nccl_bench.py $R/p5_nccl.json > $R/p5_nccl.log 2>&1

p6_ncu_shard() {  # one GPU, one kernel class, one profiled step
  local gpu=$1 tag=$2 rx=$3
  env CUDA_VISIBLE_DEVICES=$gpu MASTER_PORT=$((29620 + gpu)) BENCH_DEVICES=1 BENCH_STEPS=42 BENCH_CUDA_PROFILER=40:41 \
    BENCH_WRAP="timeout 55m $NCU --profile-from-start off --target-processes all --kernel-name regex:$rx --set roofline \
    --section Occupancy --section LaunchStats --section MemoryWorkloadAnalysis --metrics \
    sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,dram__bytes.sum,gpu__time_duration.sum \
    --nvtx -f -o $R/p6_ncu/$tag" bash $WT/jobs/bench_train.sh $R/p6_ncu/run_$tag
}
if [ $NCU_OK = 1 ]; then
  mkdir -p $R/p6_ncu
  p6_ncu_shard 0 gemm '(nvjet|gemm|cutlass|sm90|xmma)' & p6_ncu_shard 1 attn '(flash|fmha|attention)' &
  p6_ncu_shard 2 tpoi 'triton_poi' & p6_ncu_shard 3 tred '(triton_red|triton_per)' & p6_ncu_shard 4 tmm 'triton_(tem|mm|bmm)' &
  p6_ncu_shard 5 aten_ew '(elementwise|vectorized|unrolled)' & p6_ncu_shard 6 aten_red '(reduce|norm|softmax|multi_tensor)' &
  p6_ncu_shard 7 other '^(?!.*(nvjet|gemm|cutlass|sm90|xmma|flash|fmha|attention|triton|elementwise|vectorized|unrolled|reduce|norm|softmax|multi_tensor|nccl)).*' &
  phase p6_ncu wait
else
  log "p6_ncu skipped: GPU performance counters need admin (RmProfilingAdminOnly)"
fi
log "DONE $R"
du -sh $R
