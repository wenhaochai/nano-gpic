#!/bin/bash
#SBATCH -J gpic-none-probe
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:45:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Probe of the none_dealloc crash (w1239, w1204; owner 2026-11-07 02:3x: 「这个崩溃应该修复掉吧」): d640 with the new ladder's Muon recipe
# (batch 1024 = 8 x 32 x 4) for 256 optimizer steps, twice: with the Inductor C++ wrapper (as every ladder runs) and without it.
# NoneRefcount logs sys.getrefcount(None) after every micro-batch on rank 0 (runs/gpic-speed/none_probe/<arm>/none_refs.txt).
set -eo pipefail
WC=/scratch/gpfs/GROUP/USER
WT=$WC/project/gpic-wt/none_probe
for ARM in cpp1 cpp0; do
  R=$WC/runs/gpic-speed/none_probe/$ARM; rm -rf $R; mkdir -p $R/exp_ladder_d640; echo 20261106 > $R/exp_ladder_d640/shard_seed.txt
  PROBE_CPP_WRAPPER=${ARM#cpp} LADDER_WT=$WT LADDER_RUN_ROOT=$R LADDER_LAST_EVERY=256 bash $WT/jobs/train_ladder.sh d640 640 9 2 256 \
    --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr 6.4e-4 --model.optimizer.init_args.muon_lr 1.7e-3 \
    --data.train_dataset.init_args.tar_pattern 'gpic_train_{00000..07999}.tar' \
    --trainer.logger.init_args.name none_probe_$ARM --trainer.logger.init_args.id none-probe-$ARM \
    --trainer.callbacks+='{"class_path":"src.callbacks.none_refcount.NoneRefcount","init_args":{"path":"'$R'/none_refs.txt"}}' || echo "arm $ARM exit $?"
  python - $R/none_refs.txt $ARM <<'PY'
import sys
rows = [l.split() for l in open(sys.argv[1])]
ref = [int(r[-1]) for r in rows]; b = [int(r[2]) for r in rows]
n = len(ref) - 1
print(f"[probe] {sys.argv[2]}: {len(ref)} micro-batches; None refs {ref[0]} -> {ref[-1]}; mean change per micro-batch {(ref[-1] - ref[0]) / max(n, 1):+.2f}")
last = ref[len(ref) // 2:]; print(f"[probe] {sys.argv[2]}: second half, change per micro-batch {(last[-1] - last[0]) / max(len(last) - 1, 1):+.2f}")
PY
done
