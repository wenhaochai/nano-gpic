"""Tests of ladder option B (grouped, masked text). Run on vis2:
  PYTHONPATH=baselines/PixelGen CUDA_VISIBLE_DEVICES=0 python tools/test_ladder.py [--data-batches 60]
1. official path unchanged: the denoiser without y_mask equals the module before option B (ae74d35) bit for bit (fp32)
2. padding invariance: with y_mask, a sample's output does not depend on the padding after its caption (fp32)
3. text encoder (sdpa, fp32): valid positions of a padded masked batch equal each caption encoded alone
4. CFG dropout: dropped rows carry the null embedding at position 0 and a mask with one True
5. grouping on real shards: each per-GPU batch is one length-sorted chunk; mean padded text length
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PG = os.path.join(ROOT, "baselines", "PixelGen")
KW = dict(input_size=64, in_channels=3, num_groups=4, hidden_size=256, num_blocks=2, num_text_blocks=1, patch_size=16,
          txt_embed_dim=128, txt_max_length=320, bottleneck_dim=32)


def old_module():
    src = subprocess.run(["git", "-C", ROOT, "show", "ae74d35:baselines/PixelGen/src/models/transformer/JiT_T2I.py"],
                         capture_output=True, text=True, check=True).stdout
    f = tempfile.NamedTemporaryFile("w", suffix="_old_jit.py", delete=False); f.write(src); f.close()
    spec = importlib.util.spec_from_file_location("old_jit", f.name)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-batches", type=int, default=60)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ok = True
    from src.models.transformer.JiT_T2I import JiT_T2I

    # 1. official path
    torch.manual_seed(0)
    new = JiT_T2I(**{**KW, "txt_max_length": 300}).to(dev).float()
    for p in new.parameters():
        p.data.add_(torch.randn_like(p) * 0.02)
    old = old_module().JiT_T2I(**{**KW, "txt_max_length": 300}).to(dev).float()
    old.load_state_dict(new.state_dict())
    x = torch.randn(3, 3, 64, 64, device=dev); t = torch.rand(3, device=dev); y = torch.randn(3, 300, 128, device=dev)
    with torch.no_grad():
        o_new, o_old = new(x, t, y), old(x, t, y)
    same = torch.equal(o_new, o_old)
    ok &= same
    print(f"1. official path vs ae74d35 (before option B): bit-identical {same}", flush=True)

    # 2. padding invariance with masks
    torch.manual_seed(1)
    m = JiT_T2I(**KW).to(dev).float()
    for p in m.parameters():
        p.data.add_(torch.randn_like(p) * 0.02)
    lens = [5, 40, 63]
    base = torch.randn(3, 64, 128, device=dev)
    outs = {}
    for T in (64, 128, 320):
        yy = torch.randn(3, T, 128, device=dev) * 7.0          # garbage in the padded positions
        yy[:, :64] = base
        mask = torch.zeros(3, T, dtype=torch.bool, device=dev)
        for r, L in enumerate(lens):
            mask[r, :L] = True
            yy[r, L:64] = torch.randn(64 - L, 128, device=dev) * 7.0
        with torch.no_grad():
            outs[T] = m(x, t, yy, y_mask=mask)
    d1 = (outs[64] - outs[128]).abs().max().item(); d2 = (outs[64] - outs[320]).abs().max().item()
    scale = outs[64].abs().max().item()
    good = max(d1, d2) < 1e-5 * max(scale, 1.0)
    ok &= good
    print(f"2. padding invariance (T 64 vs 128 vs 320, garbage in pads): max |diff| {max(d1, d2):.2e} (scale {scale:.2e})  {'ok' if good else 'FAIL'}", flush=True)

    # 3/4. text encoder and CFG dropout
    if dev.type == "cuda":
        from src.models.conditioner.qwen3_text_encoder import Qwen3TextEncoder
        te = Qwen3TextEncoder(weight_path="Qwen/Qwen3-1.7B", embed_dim=2048, max_length=320, text_mode="masked",
                              attn_implementation="sdpa").to(dev)
        te.model.float()
        caps = ["a cat", "a red bicycle leaning against a brick wall in the late afternoon sun", "tree"]
        meta = {}
        with torch.no_grad():
            cond, unc = te(caps, meta)
        ym = meta["y_mask"]
        worst, mag = 0.0, 0.0
        for r, c in enumerate(caps):
            with torch.no_grad():
                alone, _ = te([c], {})
            L = int(ym[r].sum())
            worst = max(worst, (cond[r, :L].float() - alone[0, :L].float()).abs().max().item())
            mag = max(mag, alone[0, :L].float().abs().max().item())
        # te() returns bf16 (BaseConditioner casts), so compare relative to the magnitude at bf16 resolution (2^-8)
        good = worst / mag < 4e-3 and cond.shape[1] % 64 == 0
        ok &= good
        print(f"3. text encoder: T {cond.shape[1]}, valid positions padded vs alone max |diff| {worst:.2e} of max |value| {mag:.1f} (rel {worst/mag:.1e}, bf16 output)  {'ok' if good else 'FAIL'}", flush=True)

        from src.diffusion.base.training import BaseTrainer
        tr = BaseTrainer(null_condition_p=1.0)
        meta2 = {"y_mask": ym.clone()}
        _, c2, meta2 = tr.preproprocess(torch.zeros(3, 1), cond.clone(), unc, meta2)
        rows_ok = bool((meta2["y_mask"].sum(1) == 1).all() and meta2["y_mask"][:, 0].all())
        emb_ok = torch.equal(c2[:, 0], unc[:, 0]) and bool((c2[:, 1:] == 0).all())
        ok &= rows_ok and emb_ok
        print(f"4. CFG dropout (p=1): mask rows one-hot at 0 {rows_ok}, null embedding at 0 and zeros after {emb_ok}", flush=True)
        del te
        torch.cuda.empty_cache()

    # 5. grouping on real shards
    from src.data.dataset.gpic_dataset import GPICWebDataset
    ds = GPICWebDataset(tar_dir="/scratch/gpfs/GROUP/USER/data/gpic/train/", tar_pattern="gpic_train_{00000..00007}.tar",
                        shuffle_buffer=100, group_text_len=True, group_window=1024, text_max_length=320, pad_multiple=64)
    loader = ds.build_dataloader(batch_size=32, num_workers=4)
    Ts, sorted_ok, n = [], True, 0
    for imgs, caps, meta in loader:
        lens = meta["text_mask"].sum(1).tolist()
        sorted_ok &= lens == sorted(lens)
        Ts.append(meta["input_ids"].shape[1])
        assert imgs.shape[1:] == (3, 256, 256) and imgs.dtype == torch.float32
        n += 1
        if n >= a.data_batches:
            break
    mean_T = sum(Ts) / len(Ts)
    good = sorted_ok and mean_T < 120
    ok &= good
    hist = {T: Ts.count(T) for T in sorted(set(Ts))}
    print(f"5. grouping: {n} batches, every batch length-sorted {sorted_ok}, mean padded T {mean_T:.1f}, T histogram {hist}  {'ok' if good else 'FAIL'}", flush=True)

    # 6. shard order per epoch: one permutation per epoch shared by every rank/worker copy, new in every epoch
    from itertools import islice
    from src.data.dataset.gpic_dataset import EpochShuffledShards
    urls = [f"s{i:05d}.tar" for i in range(801)]
    world, nw, epochs = 8, 32, 3
    copies = {(r, w): EpochShuffledShards(urls, 1762100000) for r in range(world) for w in range(nw)}
    cover_ok = True
    for e in range(epochs):
        seen = []
        for (r, w), c in copies.items():
            seen += [d["url"] for d in islice(islice(iter(c), r, None, world), w, None, nw)]   # split_by_node, _by_worker
        cover_ok &= sorted(seen) == urls
    two = EpochShuffledShards(urls, 1762100000)
    a1 = [d["url"] for d in iter(two)]
    a2 = [d["url"] for d in iter(two)]
    differ = a1 != a2 and a1 != urls
    good = cover_ok and differ
    ok &= good
    print(f"6. shard order: every epoch reads each of 801 shards once over {world} ranks x {nw} workers {cover_ok}, "
          f"epoch 2 order differs from epoch 1 {a1 != a2}  {'ok' if good else 'FAIL'}", flush=True)

    # 7. grouping tail: a stream that is not a whole number of chunks yields only full chunks
    ds7 = GPICWebDataset(tar_dir="/nonexistent/", tar_pattern="x", group_text_len=True, group_window=1024)
    ds7.group_batch = 32
    stream = [{"input_ids": [0] * (1 + (i * 7919) % 300)} for i in range(1000 + 1024 + 5)]
    out = list(ds7._group(iter(stream)))
    lens = [len(s["input_ids"]) for s in out]
    chunks_sorted = all(lens[i:i + 32] == sorted(lens[i:i + 32]) for i in range(0, len(lens), 32))
    good = len(out) == (1024 + 1000 // 32 * 32) and len(out) % 32 == 0 and chunks_sorted
    ok &= good
    print(f"7. grouping tail: {len(stream)} samples in, {len(out)} out (full chunks only), chunks sorted {chunks_sorted}  {'ok' if good else 'FAIL'}", flush=True)
    print("ALL OK" if ok else "SOME CHECKS FAILED", flush=True)


if __name__ == "__main__":
    main()
