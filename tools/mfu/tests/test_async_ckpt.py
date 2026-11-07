#!/usr/bin/env python3
"""CPU test of src/plugins/async_ckpt_io.AsyncShmCheckpointIO (MFU lever F): three saves (two in flight, the third waits
for a buffer, one larger than the first buffer), teardown waits; each file loads to exactly the saved dict (tensors of
several dtypes and shapes, 0-dim, empty, bool; nested containers; Python values), its mtime is the snapshot time, no
temporary file is left. usage: python test_async_ckpt.py <scratch dir>"""
import collections, glob, os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen"))
from src.plugins.async_ckpt_io import AsyncShmCheckpointIO


def make(seed, big):
    g = torch.Generator().manual_seed(seed)
    sd = collections.OrderedDict()
    for i in range(40 if big else 20):
        sd[f"p{i}.weight"] = torch.randn(64, 48 + i, generator=g)
        sd[f"p{i}.bias"] = torch.randn(64, generator=g).bfloat16()
    sd["pos_embed"] = torch.randn(1, 256, 64, generator=g)
    sd["count"] = torch.tensor(7)
    sd["mask"] = torch.rand(5, 7, generator=g) > 0.5
    sd["empty"] = torch.empty(0, 3)
    sd["ids"] = torch.randint(0, 1000, (11,), generator=g)
    opt = {"state": {i: {"step": torch.tensor(float(seed)), "exp_avg": torch.randn(64, 10, generator=g),
                         "exp_avg_sq": torch.rand(64, 10, generator=g)} for i in range(5)},
           "param_groups": [{"lr": 1e-4, "betas": (0.9, 0.95), "params": list(range(5))}]}
    return {"state_dict": sd, "optimizer_states": [opt], "global_step": seed, "loops": {"a": [1, 2, {"b": None}]},
            "callbacks": {"X": {"best": torch.tensor(0.5)}}, "hparams": {"name": "t"}}


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main():
    d = sys.argv[1]
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(os.path.join(d, "*")):
        os.remove(f)
    io_ = AsyncShmCheckpointIO()
    cks, t_calls = [], []
    for i, big in enumerate([False, False, True]):
        ck = make(i, big)
        cks.append(ck)
        t_calls.append(time.time())
        io_.save_checkpoint(ck, os.path.join(d, f"ck{i}.ckpt"))
        for t in ck["state_dict"].values():   # training changes the tensors right after the save: the files must not see it
            if t.is_floating_point():
                t.add_(1000.0)
    io_.teardown()
    ok = True
    for i in range(3):
        got = torch.load(os.path.join(d, f"ck{i}.ckpt"), weights_only=False)
        ref = make(i, i == 2)
        ok &= same(got, ref)
        mt = os.path.getmtime(os.path.join(d, f"ck{i}.ckpt"))
        ok &= abs(mt - t_calls[i]) < 1.0
        print(f"ck{i}: content {'ok' if same(got, ref) else 'DIFFERS'}, mtime - call time {mt - t_calls[i]:+.3f} s")
    left = [f for f in os.listdir(d) if ".tmp" in f]
    ok &= not left
    print(("ok" if ok else "FAIL") + f": 3 async saves, temporary files left: {left}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
