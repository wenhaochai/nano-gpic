#!/bin/bash
#SBATCH -J gpic-mfu-probe9
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:30:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU lever E diagnosis (docs/mfu.md): why packed captions cost 2x at d640 and +3-6 ms at d1280-d1792 on one GPU (w1323).
# Eight single-GPU tasks at once (no queue tail): d640 padded and packed with torch.profiler (host syncs, kernels), packed
# with FA3 in the text blocks (w1323's code), packed with Inductor's Python wrapper (no proxy executor), packed eager
# (does the slow run match eager?), d1280 padded and packed, d640 padded control. All with FA3 in the image blocks.
# usage: probe9.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/probe9_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PIXGEN_SRC=$WT/baselines/PixelGen GPIC_MFU_ATTN=fa3
cd $WT
SB="timeout 1200 python $WT/tools/mfu/step_bench.py --steps 60"
run() { local g=$1 tag=$2; shift 2; env "$@" > $OUT/$tag.log 2>&1; echo "[$(date +%T)] $tag rc=$?" >> $OUT/summary.txt; }
run ${G[0]} prof_pad CUDA_VISIBLE_DEVICES=${G[0]} TORCHINDUCTOR_CPP_WRAPPER=1 $SB --d 640 --profile 30:33 --out $OUT/prof_pad.json &
run ${G[1]} prof_packed CUDA_VISIBLE_DEVICES=${G[1]} TORCHINDUCTOR_CPP_WRAPPER=1 $SB --d 640 --packed --profile 30:33 --out $OUT/prof_packed.json &
run ${G[2]} packed_textfa3 CUDA_VISIBLE_DEVICES=${G[2]} TORCHINDUCTOR_CPP_WRAPPER=1 GPIC_MFU_TEXT_ATTN=fa3 $SB --d 640 --packed --out $OUT/packed_textfa3.json &
run ${G[3]} packed_pywrapper CUDA_VISIBLE_DEVICES=${G[3]} TORCHINDUCTOR_CPP_WRAPPER=0 $SB --d 640 --packed --out $OUT/packed_pywrapper.json &
run ${G[4]} packed_eager CUDA_VISIBLE_DEVICES=${G[4]} $SB --d 640 --packed --no-compile --out $OUT/packed_eager.json &
run ${G[5]} d1280_pad CUDA_VISIBLE_DEVICES=${G[5]} TORCHINDUCTOR_CPP_WRAPPER=1 $SB --d 1280 --out $OUT/d1280_pad.json &
run ${G[6]} d1280_packed CUDA_VISIBLE_DEVICES=${G[6]} TORCHINDUCTOR_CPP_WRAPPER=1 $SB --d 1280 --packed --out $OUT/d1280_packed.json &
run ${G[7]} pad_ctrl CUDA_VISIBLE_DEVICES=${G[7]} TORCHINDUCTOR_CPP_WRAPPER=1 $SB --d 640 --out $OUT/pad_ctrl.json &
wait
grep -h MFU $OUT/*.log
for t in pad packed; do f=$OUT/prof_${t}_trace_rank0.json; [ -f $f ] && python $WT/tools/mfu/trace_breakdown.py $f --top 25 > $OUT/breakdown_$t.txt 2>&1; done
echo "DONE $OUT"
