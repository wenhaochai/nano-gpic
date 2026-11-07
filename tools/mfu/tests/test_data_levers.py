#!/usr/bin/env python3
"""MFU levers C, D, E end to end on the data path (CPU): the grouped, cached Lite path of configs_h100/lite_ladder.yaml on
one shard, in one process (no workers, so the grouping RNG, which mixes in the PID, is the same for both sides; shuffle
buffer 1, as wds.shuffle seeds itself from the clock), the crop RNG reseeded per side: the production code (float crops,
padded captions) against GPIC_MFU_GPU_RESIZE + UINT8 + PACK (the resize passes run by pil_exact_gpu.run under Triton's
interpreter, uint8_to_float, packed captions). The first batches must give the same float images bit for bit and the same
caption tokens. usage: python test_data_levers.py [batches]"""
import os, random, sys
os.environ.update(GPIC_REGION_CROP="1", GPIC_SHARD_SEED="7", CUDA_VISIBLE_DEVICES="", TRITON_INTERPRET="1")
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "baselines", "PixelGen"))
from src.data.dataset import gpic_dataset as G


def side(n, levers):
    G._GPU_RESIZE = G._UINT8 = G._PACK = levers
    random.seed(0)
    torch.manual_seed(0)
    ds = G.GPICWebDataset(tar_dir="/scratch/gpfs/GROUP/USER/data/gpic/train/", tar_pattern="gpic_train_00042.tar",
                          shuffle_buffer=1, group_text_len=True, group_window=1024, text_max_length=320, pad_multiple=8,
                          emb_dir="/scratch/gpfs/GROUP/USER/data/gpic/train_emb_qwen3_1.7b")
    it = iter(ds.build_dataloader(32, 0))
    res = []
    for _ in range(n):
        x, caps, meta = next(it)
        if isinstance(x, dict):
            from src.data.dataset.pil_exact_gpu import run
            x = G.uint8_to_float(run(x))
        elif x.dtype == torch.uint8:
            x = G.uint8_to_float(x)
        if "text_lens" in meta:
            toks, lens = meta["text_emb"], list(meta["text_lens"])
        else:
            lens = meta["text_mask"].sum(1).tolist()
            toks = torch.cat([meta["text_emb"][b, :n_] for b, n_ in enumerate(lens)])
        res.append({"x": x, "toks": toks, "lens": lens, "caps": list(caps)})
    return res


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    a, b = side(n, False), side(n, True)
    ok = True
    for i, (p, q) in enumerate(zip(a, b)):
        same_x = p["x"].dtype == q["x"].dtype and torch.equal(p["x"].view(torch.int32), q["x"].view(torch.int32))
        same_t = torch.equal(p["toks"].view(torch.int16), q["toks"].view(torch.int16)) and p["lens"] == q["lens"]
        ok &= same_x and same_t and p["caps"] == q["caps"]
        print(f"batch {i}: images {'same' if same_x else 'DIFFER'}, caption tokens {'same' if same_t else 'DIFFER'}, "
              f"captions {'same' if p['caps'] == q['caps'] else 'DIFFER'}, T {max(p['lens'])}", flush=True)
    print(("ok" if ok else "FAIL") + f": {len(a)} batches, production path vs levers C + D + E")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
