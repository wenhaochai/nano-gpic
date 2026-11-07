"""Every shard has a complete cache: index and file present, offsets contiguous, file size = tokens x 4096 bytes, and the
index's keys equal the tar's caption keys (from its member names only, read once per shard; every json of the 801 shards
has a caption). Training stops at a key with no entry, so this check must see every key. 16 processes, because one pass
over 801 cold tars takes about 21 min in one process and the job's GPUs are idle during the check.

python tools/check_text_cache.py <emb_dir> <tar_dir>
Exit 1 when a shard fails. Prints the totals.
"""
import glob
import os
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from webdataset.tariterators import base_plus_ext

PROCS = 16


def check_shard(args):
    """(problems, captions, tokens) of one shard."""
    t, emb_dir = args
    base = os.path.basename(t)[:-len(".tar")]
    idx, emb = os.path.join(emb_dir, base + ".idx.npz"), os.path.join(emb_dir, base + ".emb")
    if not (os.path.exists(idx) and os.path.exists(emb)):
        return [(base, "missing")], 0, 0
    bad = []
    with np.load(idx) as z:
        keys, offs, lens = z["keys"].tolist(), z["offsets"], z["lengths"]
    if len(offs) and (offs[0] != 0 or (offs[1:] != offs[:-1] + lens[:-1]).any()):
        bad.append((base, "offsets"))
    if os.path.getsize(emb) != int(lens.sum()) * 4096:
        bad.append((base, f"size {os.path.getsize(emb)} for {int(lens.sum())} tokens"))
    if lens.min() < 1 or lens.max() > 320:
        bad.append((base, f"lengths {lens.min()}..{lens.max()}"))
    with tarfile.open(t) as f:
        tar_keys = {base_plus_ext(m.name)[0] for m in f if m.name.endswith(".json")}
    if len(keys) != len(set(keys)) or set(keys) != tar_keys:
        bad.append((base, f"{len(keys)} entries, {len(tar_keys)} json files, {len(tar_keys - set(keys))} with no entry"))
    return bad, len(lens), int(lens.sum())


def main():
    emb_dir, tar_dir = sys.argv[1:3]
    tars = sorted(glob.glob(os.path.join(tar_dir, "*.tar")))
    bad, caps, tokens = [], 0, 0
    with ProcessPoolExecutor(PROCS) as ex:
        for b, c, n in ex.map(check_shard, [(t, emb_dir) for t in tars], chunksize=4):
            bad += b
            caps += c
            tokens += n
    print(f"{len(tars)} shards, {caps} captions, {tokens} tokens, mean {tokens / max(caps, 1):.1f}, {tokens * 4096 / 2**40:.3f} TiB; bad {len(bad)}")
    for b in bad[:20]:
        print("BAD", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
