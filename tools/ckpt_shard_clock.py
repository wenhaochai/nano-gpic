"""Print the shard clock of a checkpoint: '<dump time> <shard seed> <step>' (src/callbacks/shard_clock.py), for
jobs/train_ladder.sh. The checkpoint is memory-mapped, so its tensors are not read. A checkpoint without the clock (one
saved before the callback) prints '<mtime, floored> - -' and a warning on stderr.
usage: python tools/ckpt_shard_clock.py <checkpoint>
"""
import math
import os
import sys

import torch

KEY = "gpic_shard_clock"
path = sys.argv[1]
ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
clock = ck.get(KEY) if isinstance(ck, dict) else None
if not clock:
    print(f"{path}: no {KEY}; the cut falls back to the file's mtime", file=sys.stderr)
    print(f"{math.floor(os.stat(path).st_mtime)} - -")
else:
    print(f"{clock['time']:.3f} {clock.get('seed') or '-'} {clock.get('step', '-')}")
