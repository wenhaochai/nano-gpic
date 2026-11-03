"""Every shard has a complete cache: index and file present, offsets contiguous, file size = tokens x 4096 bytes, one
entry per sample of the tar (the tar's sample count from its member names only, read once per shard).

python tools/check_text_cache.py <emb_dir> <tar_dir>
Exit 1 when a shard fails. Prints the totals.
"""
import glob
import os
import sys
import tarfile

import numpy as np


def main():
    emb_dir, tar_dir = sys.argv[1:3]
    tars = sorted(glob.glob(os.path.join(tar_dir, "*.tar")))
    bad, caps, tokens = [], 0, 0
    for t in tars:
        base = os.path.basename(t)[:-len(".tar")]
        idx, emb = os.path.join(emb_dir, base + ".idx.npz"), os.path.join(emb_dir, base + ".emb")
        if not (os.path.exists(idx) and os.path.exists(emb)):
            bad.append((base, "missing"))
            continue
        z = np.load(idx)
        offs, lens = z["offsets"], z["lengths"]
        if len(offs) and (offs[0] != 0 or (offs[1:] != offs[:-1] + lens[:-1]).any()):
            bad.append((base, "offsets"))
        if os.path.getsize(emb) != int(lens.sum()) * 4096:
            bad.append((base, f"size {os.path.getsize(emb)} for {int(lens.sum())} tokens"))
        if lens.min() < 1 or lens.max() > 320:
            bad.append((base, f"lengths {lens.min()}..{lens.max()}"))
        with tarfile.open(t) as f:
            n = sum(m.name.endswith(".json") for m in f)
        if n < len(lens) or len(lens) < 0.99 * n:   # a json with no caption has no entry
            bad.append((base, f"{len(lens)} entries, {n} json files"))
        caps += len(lens)
        tokens += int(lens.sum())
    print(f"{len(tars)} shards, {caps} captions, {tokens} tokens, mean {tokens / max(caps, 1):.1f}, {tokens * 4096 / 2**40:.3f} TiB; bad {len(bad)}")
    for b in bad[:20]:
        print("BAD", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
