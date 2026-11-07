"""Full ladder (docs/ladder_plan.md section 12.4): the shard log's cut, kept inside every checkpoint.

The data workers log each shard they start (shards_started_rank<r>.txt, src/data/dataset/gpic_dataset.py EpochShuffledShards).
A resume keeps the lines up to the moment its checkpoint was taken and skips those shards. That moment must travel with the
checkpoint: the file's mtime changes when the file is copied, and a save of the largest sizes lasts up to a minute, during
which the workers keep starting shards that the checkpoint never trained (review of ladder-full, 2026-11-06, S1). This
callback stores the dump time and the run's shard seed in every checkpoint it is part of; tools/ckpt_shard_clock.py reads them
back for jobs/train_ladder.sh. It changes nothing that the job computes.
"""
import os
import time

from lightning.pytorch.callbacks import Callback

KEY = "gpic_shard_clock"


class ShardClock(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint[KEY] = {"time": time.time(), "seed": os.environ.get("GPIC_SHARD_SEED"), "step": trainer.global_step}
