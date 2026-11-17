#!/bin/bash
#SBATCH -J gpic-res-refstats
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Resolution sweep (tmp/ladder_prof/res/PLAN.md section 5, item 4): the reference statistics at 64, 128 and 512, and the evaluation at R tested.
# 1. For R in 256 64 128 512: DINOv2 features (tools/fast_eval.py --save-features, 8 GPUs) and Inception features (tools/official_inception_sharded.py,
#    one shard per GPU) of data/gpic/reference_stats/res/images/test_r<R>.npy (tools/res/crop_reference_images.py: the 50K test images of
#    test_stats.npz, the toolkit's crop at R), then tools/res/reference_stats.py build -> data/gpic/reference_stats/res/test_stats_r<R>.npz.
# 2. Check: the rebuilt 256 statistics against the shipped test_stats.npz on the embeddings of the same images (reference_stats.py compare
#    --max-rel 5e-3 --max-fd-shared 0.01: the per-image relative L2 and the FD between the two sides' mean and covariance; w2886 measured
#    6.3e-4 / 1.19e-3 and 0.00034 / 0.000022 for DINOv2 / Inception), else exit 1 before anything else is written. The shipped mu and sigma
#    come from other images (FD 7.50 / 0.669 to the mean and covariance of its own embeddings), so the FD to them is recorded, not gated.
#    Then each R against the rebuilt 256 (the same images: the FD that the resolution alone gives) into res/real_r<R>_vs_r256.json.
# 3. Evaluation smoke (EVAL_LIMIT_BATCHES=2: predict 2 batches per rank, sample check, no FD) of the smoke runs' last checkpoints
#    (runs/gpic-speed/res_smoke_1117_005820, d10 after 320 steps) at 64 p4, 128 p8, 512 p16, 512 p32 and 128 p16 through jobs/evals_gmflow.sh
#    --res, then one full evaluation (50K samples, FD against test_stats_r64.npz) at 64 p4.
# Done parts are skipped (a statistics file, a metrics.json), so a rerun does the rest.
export NCCL_NVLS_ENABLE=0   # NVIDIA/nccl #1906
set -eo pipefail
WC=/scratch/gpfs/GROUP/USER
source $WC/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/eval_res_run
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
export TMPDIR=$WC/tmp/job/${FETCHED_JOB_ID:-$SLURM_JOB_ID}; mkdir -p $TMPDIR
RS=$WC/data/gpic/reference_stats; OUT=$RS/res; IMG=$OUT/images; SHIP=$RS/test_stats.npz
[ -f $IMG/rows.npy ] || { echo "no $IMG/rows.npy: run tools/res/crop_reference_images.py first"; exit 2; }
cd $WT/baselines/PixelGen; export PYTHONPATH=$PWD
log() { echo "[$(date +%T)] $*"; }
for R in 256 64 128 512; do
  F=$OUT/test_stats_r$R.npz
  if [ ! -f $F ]; then
    log "features r$R"
    python -m torch.distributed.run --standalone --nproc_per_node 8 $WT/tools/fast_eval.py $IMG/test_r$R.npy $SHIP $TMPDIR/dino_r$R.json \
      --workers 6 --save-features $TMPDIR/dino_r$R.npy > $TMPDIR/dino_r$R.log 2>&1 || { tail -20 $TMPDIR/dino_r$R.log; exit 1; }
    pids=()
    for s in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$s XDG_CACHE_HOME=$WC/cache/gpic_eval_xdg TF_FORCE_GPU_ALLOW_GROWTH=true $WC/envs/gpic-eval-tf/bin/python \
        $WT/tools/official_inception_sharded.py features $IMG/test_r$R.npy $s 8 $TMPDIR/inc_r${R}_$s.npy > $TMPDIR/inc_r${R}_$s.log 2>&1 & pids+=($!)
    done
    for p in "${pids[@]}"; do wait $p || { tail -20 $TMPDIR/inc_r${R}_*.log; exit 1; }; done
    python $WT/tools/res/reference_stats.py build $TMPDIR/dino_r$R.npy $IMG/rows.npy $F.partial.npz $(for s in 0 1 2 3 4 5 6 7; do echo $TMPDIR/inc_r${R}_$s.npy; done)
    mv $F.partial.npz $F
  fi
  if [ $R = 256 ]; then
    log "check: rebuilt 256 against the shipped embeddings"
    python $WT/tools/res/reference_stats.py compare $F $SHIP --max-rel 5e-3 --max-fd-shared 0.01 --b-self | tee $OUT/check_r256.json \
      || { echo "the rebuilt 256 embeddings do not match the shipped ones"; exit 1; }
  else
    python $WT/tools/res/reference_stats.py compare $F $OUT/test_stats_r256.npz > $OUT/real_r${R}_vs_r256.json
  fi
done
log "evaluation smoke"
SM=$WC/runs/gpic-speed/res_smoke_1117_005820 ER=$WC/runs/gpic-speed/res_eval_smoke
fail=0
for RP in 64:4 128:8 512:16 512:32 128:16; do
  R=${RP%%:*} P=${RP#*:}; RUN=$ER/r${R}p$P
  mkdir -p $RUN/exp_ladder_d768; [ -e $RUN/exp_ladder_d768/ms-step=000320.ckpt ] || ln -s $SM/r${R}p$P/exp_ladder_d768/last.ckpt $RUN/exp_ladder_d768/ms-step=000320.ckpt
  if EVAL_LIMIT_BATCHES=2 bash $WT/jobs/evals_gmflow.sh 0 d768 768 10 2 $RUN --wt eval_res_run --res $RP 320:euler:50; then
    log "smoke r${R}p$P: $(cat $RUN/exp_ladder_d768/eval_000320_euler_calls50_official/sample_check.txt) $(cat $RUN/exp_ladder_d768/eval_000320_euler_calls50_official/phase_times.json)"
    grep -q "^samples ([0-9]*, $R, $R, 3) uint8" $RUN/exp_ladder_d768/eval_000320_euler_calls50_official/sample_check.txt \
      || { log "smoke r${R}p$P: the samples are not (N, $R, $R, 3) uint8"; fail=1; }   # the samples themselves leave with the job's node memory
    mv $RUN/exp_ladder_d768/eval_000320_euler_calls50_official $RUN/exp_ladder_d768/smoke_limit2_$(date +%H%M%S)   # the full evaluation below takes the name
  else
    log "smoke r${R}p$P FAILED"; tail -30 $RUN/exp_ladder_d768/eval_000320_euler_calls50_official/predict.log 2>/dev/null; fail=1
  fi
done
log "full evaluation r64p4"
bash $WT/jobs/evals_gmflow.sh 0 d768 768 10 2 $ER/r64p4 --wt eval_res_run --res 64:4 320:euler:50 || fail=1
cat $ER/r64p4/exp_ladder_d768/eval_000320_euler_calls50_official/metrics.json $ER/r64p4/exp_ladder_d768/eval_000320_euler_calls50_official/metrics_inception.json 2>/dev/null || fail=1
log "done, fail=$fail"
exit $fail
