#!/usr/bin/env python3
"""MFU levers C, D, E end to end on the data path (CPU): the grouped, cached Lite path of configs_h100/lite_ladder.yaml
on one shard with 2 workers and a fixed seed, run as the production code (float crops, padded captions) and with
GPIC_MFU_GPU_RESIZE=1 GPIC_MFU_UINT8=1 GPIC_MFU_PACK=1 (resize passes run by pil_exact_gpu.run under Triton's interpreter,
uint8_to_float, packed captions). The first batches must give the same float images bit for bit and the same caption
tokens. usage: python test_data_levers.py [batches]   (it runs both sides in subprocesses)"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "..", "baselines", "PixelGen")


def side(out, n):
    import random
    import torch
    sys.path.insert(0, SRC)
    from src.data.dataset.gpic_dataset import GPICWebDataset, uint8_to_float
    torch.manual_seed(0)
    random.seed(0)
    ds = GPICWebDataset(tar_dir="/scratch/gpfs/GROUP/USER/data/gpic/train/", tar_pattern="gpic_train_00042.tar",
                        shuffle_buffer=100, group_text_len=True, group_window=1024, text_max_length=320, pad_multiple=8,
                        emb_dir="/scratch/gpfs/GROUP/USER/data/gpic/train_emb_qwen3_1.7b")
    it = iter(ds.build_dataloader(32, 2))
    res = []
    for _ in range(n):
        x, caps, meta = next(it)
        if isinstance(x, dict):
            from src.data.dataset.pil_exact_gpu import run
            x = uint8_to_float(run(x))
        elif x.dtype == torch.uint8:
            x = uint8_to_float(x)
        if "text_lens" in meta:
            toks, lens = meta["text_emb"], meta["text_lens"]
        else:
            lens = meta["text_mask"].sum(1).tolist()
            toks = torch.cat([meta["text_emb"][b, :n_] for b, n_ in enumerate(lens)])
        res.append({"x": x, "toks": toks, "lens": list(lens), "caps": list(caps)})
    ds.stop_workers()
    torch.save(res, out)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    tmp = os.environ.get("TMPDIR", "/tmp")
    base = dict(os.environ, GPIC_REGION_CROP="1", GPIC_SHARD_SEED="7", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    lev = dict(base, GPIC_MFU_GPU_RESIZE="1", GPIC_MFU_UINT8="1", GPIC_MFU_PACK="1", TRITON_INTERPRET="1")
    outs = []
    for tag, env in (("prod", base), ("levers", lev)):
        out = os.path.join(tmp, f"test_data_levers_{tag}_{os.getpid()}.pt")
        subprocess.run([sys.executable, __file__, "--side", out, str(n)], env=env, check=True)
        outs.append(out)
    import torch
    a, b = (torch.load(o, weights_only=False) for o in outs)
    ok = True
    for i, (p, q) in enumerate(zip(a, b)):
        same_x = p["x"].dtype == q["x"].dtype and torch.equal(p["x"].view(torch.int32), q["x"].view(torch.int32))
        same_t = torch.equal(p["toks"].view(torch.int16), q["toks"].view(torch.int16)) and p["lens"] == q["lens"]
        ok &= same_x and same_t and p["caps"] == q["caps"]
        print(f"batch {i}: images {'same' if same_x else 'DIFFER'}, caption tokens {'same' if same_t else 'DIFFER'}, "
              f"T {max(p['lens'])}", flush=True)
    for o in outs:
        os.remove(o)
    print(("ok" if ok else "FAIL") + f": {len(a)} batches, production path vs levers C + D + E")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--side":
        side(sys.argv[2], int(sys.argv[3]))
    else:
        main()
