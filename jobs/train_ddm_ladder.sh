#!/bin/bash
#SBATCH -J gpic-sweep
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=03:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# the cycles preemption loop skips a guest that ends within 10 min (controller 2026-11-19): a helper writes $WQ_ROOT/jobs/<n>/progress.json
# from this job's log every minute and ends with this shell
[ -n "${WQ_JOB_ID:-}" ] && { nice -n 19 python3 -I /scratch/gpfs/GROUP/USER/tmp/ladder_prof/wq_progress_sidecar.py --job "$WQ_JOB_ID" --pid $$ > /dev/null 2>&1 & }
# Non-EMA hyper-parameter sweeps for the ladder playbook (owner 2026-11-07: no EMA, 「按方案排」; 21:5x: Muon for the matrices, AdamW for
# embeddings and scalars, 「按以前的分法」): a linear cooldown over the last <cooldown %> of the steps to an absolute 3e-5, 1% warmup, raw weights
# evaluated afterwards. Weight decay only on the Muon matrices. From scratch, resumable. Code: gpic-wt/muon_sweep_run (ladder-sweep c6fdbf6).
# Owner 2026-11-07 21:3x (all required): linear decay, an absolute LR floor of 3e-5, warmup = 1% of the run.
# _ar2 (2026-11-09, sweep XL2) = _ar plus: the shape 1408 x (15+3) under FSDP2 (DDP would need about 71 of 79.6 GiB); optional main.py arguments after
# the 11th argument (a smoke test passes --trainer.max_steps); SWEEP_RUN_ROOT moves the run directory (a smoke test must not leave a last.ckpt
# in muon_sweep/<tag>, where the real run would resume from it). With 11 arguments and no SWEEP_RUN_ROOT it runs exactly what _ar runs.
# usage: train_sweep.sh <tag> <d> <nb> <nt> <steps> <muon lr> <adamw lr> <muon weight decay> <data: full|lite|s80|s8> <global batch> <cooldown %> [main.py args]
# train_ddm_ladder.sh (2026-11-22; owner popup 02:0x via the controller, option A, and 03:xx: P1 at d10 first) = train_gmflow_ladder.sh with three changes:
# (1) the tag is ddm-..., and the run is the distributional denoiser (docs/ddm.md; configs_h100/ddm_train.yaml: iDDM's recipe, 4 samples per image forked before
#     the last 2 blocks, 256 noise channels per token, the energy score per token with iDDM's linear schedules, no caption dropout, the point model's time
#     sampling); everything else is the final ladder's recipe, argument for argument;
# (2) code: the frozen worktree gpic-wt/ddm_run (branch ddm = gmflow-ladder 14fa39c + the distributional denoiser);
# (3) runs live in runs/gpic-speed/ddm_ladder/<tag>. Steps are FLOP-matched to the point run (tools/ddm_ladder/steps.py), so pass the list row as it is.
#     The micro-batch rule below counts the 2 per-sample blocks 4 times, at their width d + 256.
export GPIC_MFU_NO_EMA=1   # owner 2026-11-08 「ema是一定不能用」: no EMA update; checkpoints hold the raw weights in the EMA slot too
export GPIC_LAUNCH_RANKS=1 GPIC_LAUNCH_SKIP_EMA_COPY=1   # launch levers B (all 8 ranks start at once) and G1 (no second copy into the EMA slot): owner popup 2026-11-10 12:49, only for jobs submitted from then on (the queued snapshots do not have this line)
export LD_PRELOAD=/scratch/gpfs/GROUP/USER/envs/nccl/2.27.7/libnccl.so.2   # NCCL 2.27.7 with NVLS on: correct (gpic-28 w2147, w2171) and 1-2% faster than 2.26.2 with NVLS off (w2148); owner OK via the controller 2026-11-08. A job without this line must keep NCCL_NVLS_ENABLE=0.
export NCCL_DEBUG=VERSION   # each rank prints the NCCL version it loaded (expect 2.27.7); torch.cuda.nccl.version() shows the compiled 2.26.2
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
[ $# -ge 11 ] || { echo "usage: train_sweep.sh <tag> <d> <nb> <nt> <steps> <muon lr> <adamw lr> <muon wd> <data> <global batch> <cooldown %> [main.py args]"; exit 2; }
TAG=$1 WD=$2 NB=$3 NT=$4 S=$5 MLR=$6 ALR=$7 WDK=$8 DATA=$9 GB=${10} CDP=${11}
[[ $TAG =~ ^ddm-d[0-9]+- ]] || { echo "the tag must start with ddm-d<N>-"; exit 2; }
[ $NB -ge 3 ] || { echo "the per-sample part is the last 2 of at least 3 blocks"; exit 2; }
[[ $CDP =~ ^[0-9]+$ && $CDP -ge 5 && $CDP -le 90 ]] || { echo "bad cooldown %"; exit 2; }
[[ $GB =~ ^(256|512|1024|2048|4096)$ ]] || { echo "bad global batch"; exit 2; }
WU=$(( S / 100 )); [ $WU -ge 1 ] || WU=1
SIZE=d$WD
[[ $TAG =~ ^[0-9a-z.-]+$ && $S =~ ^[0-9]+$ && $((S % 32)) -eq 0 ]] || { echo "bad tag or steps"; exit 2; }
CD=$(( (S * CDP / 100) / 32 * 32 ))
case $DATA in full) PAT='gpic_train_{00000..07999}.tar' ;; lite) PAT='gpic_train_{00000..00800}.tar' ;;
  s80) PAT='gpic_train_{00000..00079}.tar' ;; s8) PAT='gpic_train_{00000..00007}.tar' ;; *) echo "bad data"; exit 2 ;; esac
SEED=20261106
WT=${SWEEP_WT:-$WC/project/gpic-wt/ddm_run}   # ddm (see the header); the point runs' note: sweep-noema-cap 8e1b9ef = sweep-noema dc8c02d + gpic-28's shared-bucket pool cap 40 and prefetch 2 (2e4ee4b); since 2026-11-10 launch4_cap_run be47945 = onelast_cap_run a3972a9 + launch levers A1, E, F + the cut fix (owner popup 2026-11-10 12:49)
[ -e "$WT/.git" ] || { echo "no worktree $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
grep -q 'xi_mode == "cat"' $WT/baselines/PixelGen/src/models/transformer/JiT_T2I.py && [ -f $WT/baselines/PixelGen/configs_h100/ddm_train.yaml ] || { echo "$WT has no distributional-denoiser code"; exit 2; }
# FSDP2 only for the ladder's wide sizes; every other shape is DDP at 128 images per GPU
case "$WD:$NB" in 1280:14) MB=128 FS=1 G=7 ;; 1024:24) MB=64 FS=0 G=1 ;; 896:32) MB=64 FS=0 G=1 ;; 1408:15) MB=128 FS=1 G=1 ;; 1536:*) MB=128 FS=1 G=1 ;; 1792:*) MB=64 FS=1 G=1 ;; *) MB=128 FS=0 G=1 ;; esac
# _ar3 (2026-11-11, sweep ARD): a shape with no case above halves the micro-batch while MB x width x (nb + nt) > 2.1M (tpp_plan.py's limit;
# 896 x 32 at 64 = 1.84M ran): 512 x 37, 448 x 49, 384 x 68 at 64; 768 x 44, 640 x 64, 512 x 102 at 32 (accumulation 2 and 4).
# ddm: the last 2 blocks run 4 times at width WD + 256, so they count as ceil(8 (WD + 256) / WD) blocks in the rule (d10: 8 + 11 + 2 -> 128 x 768 x 21 = 2.06M)
NBE=$(( NB - 2 + (8 * (WD + 256) + WD - 1) / WD ))
case "$WD:$NB" in 1280:14|1024:24|896:32|1408:15|1536:*|1792:*) ;; *) while [ $((MB * WD * (NBE + NT))) -gt 2100000 ]; do MB=$((MB / 2)); done ;; esac
[ $((GB / 8)) -ge $MB ] || MB=$((GB / 8))   # small global batches: fewer images per GPU, no accumulation
export GPIC_MFU_FSDP=$FS GPIC_MFU_FSDP_GROUP=$G GPIC_MFU_FSDP_NOSYNC=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_MFU_PATCHIFY=$FS GPIC_MFU_NS_FUSED=1 GPIC_MFU_LOCAL_OPT=$FS GPIC_MFU_MUON_ADD=1
export LADDER_WT=$WT LADDER_RUN_ROOT=${SWEEP_RUN_ROOT:-$WC/runs/gpic-speed/ddm_ladder/$TAG} LADDER_LAST_EVERY=$((S / 8))
mkdir -p $LADDER_RUN_ROOT/exp_ladder_$SIZE
[ -s $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt ] || echo $SEED > $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt
exec bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $S \
  --config configs_h100/muon_ladder.yaml --config configs_h100/ddm_train.yaml --model.denoiser.init_args.xi_block $((NB - 2)) --model.optimizer.init_args.lr $ALR --model.optimizer.init_args.muon_lr $MLR \
  --model.optimizer.init_args.weight_decay 0 --model.optimizer.init_args.muon_weight_decay $WDK \
  --model.lr_scheduler.init_args.num_warmup_steps $WU --model.lr_scheduler.init_args.num_cooldown_steps $CD --model.lr_scheduler.init_args.max_steps $S --model.lr_scheduler.init_args.min_lr 3e-5 \
  --data.train_batch_size $MB --trainer.accumulate_grad_batches $((GB / 8 / MB)) --data.train_dataset.init_args.shared_bucket true \
  --data.train_dataset.init_args.tar_pattern "$PAT" \
  --trainer.logger.init_args.name sweep_${TAG}_$SIZE --trainer.logger.init_args.id sweep-$TAG-$SIZE "${@:12}"
