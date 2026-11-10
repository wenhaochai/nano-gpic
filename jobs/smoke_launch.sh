#!/bin/bash
#SBATCH -J gpic-smoke-launch
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Smoke of the launch levers of jobs/train_ladder.sh (gpic-28 2026-11-09, logs/mfu_infra_scan_1109b.yaml item 6) before
# they go to production: the gate beside main.py's start, the vector-set memo and the thread count from the start.
# Each launch goes through the real launcher on 8 GPUs with the sweep path's arguments, once with the three levers
# off and once with them on, off arms first and last so a drift of the node shows between them:
#  d640    width 640 (DDP, host-bound): off, on with an empty memo (every rank asks the compiler and stores the answer),
#          on with the memo filled, off. 384 steps each: the levers set each rank's thread count before fit, where
#          Lightning sets it, and the gate runs beside the start, so the step time is part of the test.
#  d1536   width 1536 (FSDP2): off, on, off. 128 steps each, and last.ckpt at step 128.
#  d1536r  a resume at width 1536 from the first d1536 arm's last.ckpt, in that run directory (no new last.ckpt, so all
#          three start from the same file): off, on, off, to step 256.
# src/callbacks/launch_fingerprint.py prints the state every rank starts its first step from (weights, EMA slot,
# optimizer, scheduler, random streams, first batches) and the times, also of the EMA slot's second copy in
# configure_model (4.2 s at width 1536 in the w2441 profile; no lever yet); tools/mfu/check_launch.py compares them.
# Result: SMOKE_LAUNCH PASS | FAIL <reasons>; the checkpoints are deleted on PASS and kept on FAIL. About 30 min
# (130 s a width-640 arm, 200 s a width-1536 arm); an arm that would pass the job's hour is not started (FAIL <arm>-no-time).
# Plan "ranks" (lever B, GPIC_LAUNCH_RANKS=1: the launcher starts every rank at once; the plan above is "levers"): the
# first three levers are on in every arm (they passed the plan above), the base arms differ from the test arm only in
# lever B: at each of the three settings base, B, base (9 launches, about 27 min). The first arm fills the memo.
# usage: smoke_launch.sh [worktree under gpic-wt, default launch2_cap_run] [levers | ranks, default levers]
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
WT=$WC/project/gpic-wt/${1:-launch2_cap_run}
PLAN=${2:-levers}
case $PLAN in levers|ranks) ;; *) echo "plan must be levers or ranks"; exit 2 ;; esac
[ -e "$WT/.git" ] || { echo "no worktree $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
echo "code $(basename $WT) $(git -C $WT rev-parse --short HEAD) on $(hostname); plan $PLAN; OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} nproc=$(nproc)"
T_JOB=$(date +%s)
STAMP=$(date +%m%d_%H%M)
OUT=$WC/runs/gpic-speed/mfu_probe/launch_$STAMP; mkdir -p $OUT
MAN=$OUT/manifest.txt; : > $MAN
FAIL=""
run() {   # <arm> <run tag> <lever> <d> <nb> <nt> <mb> <fsdp> <max steps> <last every> <ranks>: one launch of the sweep path
  local arm=$1 tag=$2 lever=$3 d=$4 nb=$5 nt=$6 MB=$7 FS=$8 S=$9 LAST=${10} root=$OUT/$2 size=d$4 SCH=4096
  if [ "${11:-0}" = 1 ]; then export GPIC_LAUNCH_RANKS=1; else unset GPIC_LAUNCH_RANKS; fi
  mkdir -p $root/exp_ladder_$size; [ -s $root/exp_ladder_$size/shard_seed.txt ] || echo 20261106 > $root/exp_ladder_$size/shard_seed.txt
  if [ $lever = off ]; then export GPIC_LAUNCH_GATE_BG=0 GPIC_LAUNCH_VEC_ISA_MEMO=0 GPIC_LAUNCH_OMP=0
  else unset GPIC_LAUNCH_GATE_BG GPIC_LAUNCH_VEC_ISA_MEMO GPIC_LAUNCH_OMP; export GPIC_VEC_ISA_MEMO=$OUT/vec_isa_memo; fi
  GPIC_MFU_NO_EMA=1 LD_PRELOAD=$WC/envs/nccl/2.27.7/libnccl.so.2 \
  GPIC_MFU_FSDP=$FS GPIC_MFU_FSDP_GROUP=1 GPIC_MFU_FSDP_NOSYNC=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_MFU_PATCHIFY=$FS \
  GPIC_MFU_NS_FUSED=1 GPIC_MFU_LOCAL_OPT=$FS GPIC_MFU_MUON_ADD=1 LADDER_WT=$WT LADDER_RUN_ROOT=$root LADDER_LAST_EVERY=$LAST \
  LADDER_PARTIAL_OK=1 GPIC_LAUNCH_FP=1 BENCH_OUT=$OUT/$arm.bench \
  exec bash $WT/jobs/train_ladder.sh $size $d $nb $nt $SCH \
    --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr 0.00064 --model.optimizer.init_args.muon_lr 1.36e-2 \
    --model.optimizer.init_args.weight_decay 0 --model.optimizer.init_args.muon_weight_decay 0.002752 \
    --model.lr_scheduler.init_args.num_warmup_steps $((SCH / 100)) --model.lr_scheduler.init_args.num_cooldown_steps $(( (SCH / 5) / 32 * 32 )) \
    --model.lr_scheduler.init_args.max_steps $SCH --model.lr_scheduler.init_args.min_lr 3e-5 \
    --data.train_batch_size $MB --trainer.accumulate_grad_batches $((1024 / 8 / MB)) --data.train_dataset.init_args.shared_bucket true \
    --data.train_dataset.init_args.tar_pattern 'gpic_train_{00000..07999}.tar' \
    --trainer.logger.init_args.name smoke_launch_$arm --trainer.logger.init_args.id smoke-launch-$arm-$STAMP \
    --trainer.max_steps $S
}
RANKS=0 ROLE="" GAIN=""   # set before a launch of the plan "ranks": lever B, base or test arm, the seconds the test arm must gain
launch() {   # <arm> <group> <lever> <memo> <start step> <run tag> <launch number in the run directory> <d> <nb> <nt> <mb> <fsdp> <max steps> <last every>
  local arm=$1 group=$2 lever=$3 memo=$4 step=$5 tag=$6 n=$7; shift 7
  local t0=$(date +%s)
  local lim=$(( 3240 - (t0 - T_JOB) )); [ $lim -le 900 ] || lim=900   # every arm and the check end inside the job's hour
  if [ $lim -lt 300 ]; then echo "== $arm: not started, $(( t0 - T_JOB )) s of the job's hour are gone"; FAIL="$FAIL $arm-no-time"; return; fi
  echo "== $arm (levers $lever, memo $memo): from step $step to $6 ($(date +%T), limit $lim s)"
  ( export -f run; export OUT WC WT STAMP; timeout --kill-after=120 ${lim}s bash -c "run $arm $tag $lever $* $RANKS" ) > $OUT/$arm.log 2>&1
  local rc=$? wall=$(( $(date +%s) - t0 ))
  echo "$arm: rc $rc in $wall s; $(grep -a -h 'resuming from' $OUT/$arm.log | head -1)"
  grep -a -h -o -E '\[gate\] [^[]*|unhealthy GPU.*' $OUT/$arm.log | head -3 | sed "s/^/$arm: /"
  echo "arm name=$arm group=$group lever=$lever memo=$memo step=$step log=$OUT/$arm.log run=$OUT/$tag/exp_ladder_d$1 n=$n bench=$OUT/$arm.bench rc=$rc wall=$wall ranks=$RANKS${ROLE:+ role=$ROLE}${GAIN:+ gain=$GAIN}" >> $MAN
  [ $rc = 0 ] || FAIL="$FAIL $arm-rc$rc"
}
state() { find $1 -maxdepth 1 -name "last*.ckpt" -printf '%f %s %T@ %i\n' | sort; }   # name, size, mtime, inode

if [ $PLAN = levers ]; then
#      arm         group  lever memo step run tag     n    d  nb nt  mb fs steps last
launch d640_off1   d640   off   off    0  d640_off1   1  640  9  2 128  0  384 4096
launch d640_miss   d640   on    miss   0  d640_miss   1  640  9  2 128  0  384 4096
launch d640_hit    d640   on    hit    0  d640_hit    1  640  9  2 128  0  384 4096
launch d640_off2   d640   off   off    0  d640_off2   1  640  9  2 128  0  384 4096
echo "memo: $(ls $OUT/vec_isa_memo 2> /dev/null | wc -l) files in $OUT/vec_isa_memo"
launch d1536_off1  d1536  off   off    0  d1536_off1  1 1536 16  4 128  1  128  128
launch d1536_hit   d1536  on    hit    0  d1536_hit   1 1536 16  4 128  1  128  128
launch d1536_off2  d1536  off   off    0  d1536_off2  1 1536 16  4 128  1  128  128
R=$OUT/d1536_off1/exp_ladder_d1536
state $R > $OUT/parent.before; cat $OUT/parent.before
[ "$(wc -l < $OUT/parent.before)" = 1 ] || FAIL="$FAIL no-parent-checkpoint"
launch d1536r_off1 d1536r off   off  128  d1536_off1  2 1536 16  4 128  1  256 4096
launch d1536r_hit  d1536r on    hit  128  d1536_off1  3 1536 16  4 128  1  256 4096
launch d1536r_off2 d1536r off   off  128  d1536_off1  4 1536 16  4 128  1  256 4096
else   # plan ranks: lever B against the launch with the first three levers
b() { RANKS=$1 ROLE=$2 GAIN=$3; }
#                    arm         group  lever memo step run tag     n    d  nb nt  mb fs steps last
b 0 base "";  launch d640_base1  d640   on    miss   0  d640_base1  1  640  9  2 128  0  384 4096
b 1 test 3;   launch d640_b      d640   on    hit    0  d640_b      1  640  9  2 128  0  384 4096
b 0 base "";  launch d640_base2  d640   on    hit    0  d640_base2  1  640  9  2 128  0  384 4096
echo "memo: $(ls $OUT/vec_isa_memo 2> /dev/null | wc -l) files in $OUT/vec_isa_memo"
b 0 base "";  launch d1536_base1 d1536  on    hit    0  d1536_base1 1 1536 16  4 128  1  128  128
b 1 test 5;   launch d1536_b     d1536  on    hit    0  d1536_b     1 1536 16  4 128  1  128  128
b 0 base "";  launch d1536_base2 d1536  on    hit    0  d1536_base2 1 1536 16  4 128  1  128  128
R=$OUT/d1536_base1/exp_ladder_d1536
state $R > $OUT/parent.before; cat $OUT/parent.before
[ "$(wc -l < $OUT/parent.before)" = 1 ] || FAIL="$FAIL no-parent-checkpoint"
b 0 base "";  launch d1536r_base1 d1536r on   hit  128  d1536_base1 2 1536 16  4 128  1  256 4096
b 1 test 5;   launch d1536r_b     d1536r on   hit  128  d1536_base1 3 1536 16  4 128  1  256 4096
b 0 base "";  launch d1536r_base2 d1536r on   hit  128  d1536_base1 4 1536 16  4 128  1  256 4096
fi
state $R > $OUT/parent.after
cmp -s $OUT/parent.before $OUT/parent.after || { FAIL="$FAIL parent-checkpoint-changed"; cat $OUT/parent.after; }

source $WC/project/gpic/jobs/env.sh
echo "== check ($(date +%T))"
OMP_NUM_THREADS=16 python $WT/tools/mfu/check_launch.py $MAN 2>&1 | tee $OUT/check.log
[ ${PIPESTATUS[0]} = 0 ] || FAIL="$FAIL check"

if [ -z "$FAIL" ]; then
  find $OUT -name "*.ckpt" -delete
  echo "SMOKE_LAUNCH PASS $OUT"
else
  echo "SMOKE_LAUNCH FAIL$FAIL $OUT (checkpoints kept: $(du -sh $OUT | cut -f1))"
  exit 1
fi
