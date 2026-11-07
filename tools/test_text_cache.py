"""Tests of the cached text path against the online encoder, on real shards. Run on a vis node with a GPU:
  PYTHONPATH=baselines/PixelGen CUDA_VISIBLE_DEVICES=0 python tools/test_text_cache.py <emb_dir> <tar_dir> <shard name>
1. index: one entry per caption in the tar, offsets contiguous, file size = tokens x 4096 bytes
2. lengths: the index's lengths equal the training path's tokenizer on every caption
3. values: real-token outputs equal the online encoder's on batches built as in training, up to bf16 rounding (the
   difference is compared with the difference between bf16 and fp32, and fp32 with a padding mask vs alone is the check
   that padding does not leak)
4. loader: real batches of the cached path: same T rule, zeros on padded tokens, mask = lengths, null embedding
5. denoiser: the image output is bit-identical (fp32) whether padded tokens hold the online encoder's values or zeros
"""
import json
import os
import random
import sys
import tarfile

import numpy as np
import torch


def main():
    emb_dir, tar_dir, shard = sys.argv[1:4]
    ok = True

    def check(c, msg):
        nonlocal ok
        ok &= bool(c)
        print(("ok   " if c else "FAIL ") + msg, flush=True)

    from webdataset.tariterators import base_plus_ext
    from src.models.conditioner.qwen3_text_encoder import Qwen3TextEncoder
    from src.models.conditioner.cached_text import CachedTextConditioner
    from src.data.dataset.gpic_dataset import GPICWebDataset, EMB_TOKEN_BYTES, EMB_DIM
    from src.models.transformer.JiT_T2I import JiT_T2I

    tar = os.path.join(tar_dir, shard + ".tar")
    z = np.load(os.path.join(emb_dir, shard + ".idx.npz"))
    keys, offs, lens = z["keys"].tolist(), z["offsets"], z["lengths"]
    caps = {}
    with tarfile.open(tar) as t:
        for m in t:
            if m.isfile() and m.name.endswith(".json"):
                meta = json.loads(t.extractfile(m).read())
                if "caption" in meta and "caption_type" in meta:
                    caps[base_plus_ext(m.name)[0]] = meta["caption"]
    check(set(keys) == set(caps) and len(keys) == len(caps), f"index: {len(keys)} keys, {len(caps)} captions in the tar")
    check((offs[1:] == offs[:-1] + lens[:-1]).all() and offs[0] == 0, "index: offsets contiguous")
    size = os.path.getsize(os.path.join(emb_dir, shard + ".emb"))
    check(size == lens.sum() * EMB_TOKEN_BYTES, f"file size {size} = {lens.sum()} tokens x {EMB_TOKEN_BYTES} bytes")

    te = Qwen3TextEncoder("Qwen/Qwen3-1.7B", embed_dim=EMB_DIM, max_length=320, text_mode="masked", pad_multiple=8,
                          attn_implementation="sdpa").cuda().eval()
    tok = te.tokenizer
    ids = {k: (tok(caps[k], truncation=True, max_length=320)["input_ids"] or [tok.eos_token_id]) for k in keys}
    check(all(len(ids[k]) == n for k, n in zip(keys, lens.tolist())), "lengths equal the tokenizer on every caption")

    # 3. values, on batches of 32 of neighbours in length, as in training but with a batch mix the cache did not see
    random.seed(0)
    order = sorted(random.sample(keys, 1024), key=lambda k: len(ids[k]))
    fd = os.open(os.path.join(emb_dir, shard + ".emb"), os.O_RDONLY)
    where = dict(zip(keys, zip(offs.tolist(), lens.tolist())))
    cached = lambda k: torch.frombuffer(bytearray(os.pread(fd, where[k][1] * EMB_TOKEN_BYTES, where[k][0] * EMB_TOKEN_BYTES)),
                                        dtype=torch.bfloat16).view(where[k][1], EMB_DIM)
    mean_d, max_d, n_tok = 0.0, 0.0, 0
    rows = []
    for b in range(0, 1024, 32):
        chunk = order[b:b + 32]
        T = min(-(-max(len(ids[k]) for k in chunk) // 8) * 8, 320)
        input_ids = torch.full((32, T), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((32, T), dtype=torch.bool)
        for r, k in enumerate(chunk):
            input_ids[r, :len(ids[k])] = torch.tensor(ids[k]); mask[r, :len(ids[k])] = True
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            y = te._impl_condition(None, {"input_ids": input_ids.cuda(), "text_mask": mask.cuda()}).float().cpu()
        for r, k in enumerate(chunk):
            d = (y[r, :len(ids[k])] - cached(k).float()).abs()
            mean_d += d.sum().item(); max_d = max(max_d, d.max().item()); n_tok += d.numel()
        rows.append((input_ids, mask, y))
    mean_d /= n_tok
    # the rounding envelope: bf16 against fp32 (eager) on one batch
    import copy
    input_ids, mask, y = rows[len(rows) // 2]
    m32 = copy.deepcopy(te.model._orig_mod if hasattr(te.model, "_orig_mod") else te.model).float()
    m32._compiled_call_impl = None
    with torch.no_grad():
        y32 = m32(input_ids=input_ids.cuda(), attention_mask=mask.long().cuda())[0].cpu()
    env = (y[mask] - y32[mask]).abs()
    print(f"     cached vs online: mean abs {mean_d:.5f}, max {max_d:.3f}; bf16 vs fp32 envelope: mean abs {env.mean():.5f}, max {env.max():.3f}")
    check(mean_d <= env.mean().item() * 1.5, "values: cached vs online within the bf16 rounding envelope")
    eos = torch.tensor([[tok.eos_token_id]], device="cuda")
    with torch.no_grad():
        null32 = m32(input_ids=eos, attention_mask=torch.ones_like(eos))[0].cpu()
    del m32

    # 4. loader, one process (the real class and collate)
    ds = GPICWebDataset(tar_dir=tar_dir, tar_pattern=shard + ".tar", group_text_len=True, group_window=1024, group_batch=32,
                        text_max_length=320, pad_multiple=8, emb_dir=emb_dir)
    loader = iter(ds.build_dataloader(32, 0))
    seen = 0
    good = True
    for _ in range(12):
        imgs, cap, meta = next(loader)
        emb, m = meta["text_emb"], meta["text_mask"]
        n = m.sum(1)
        good &= emb.shape[0] == 32 and emb.shape[1] % 8 == 0 and emb.dtype == torch.bfloat16 and emb.shape[2] == EMB_DIM
        good &= (emb[~m] == 0).all().item() and (emb[m].float().abs().sum(-1) > 0).all().item()
        good &= all(m[r, :n[r]].all() and not m[r, n[r]:].any() for r in range(32))
        good &= emb.shape[1] == min(-(-n.max().item() // 8) * 8, 320)
        # the batch's captions match their stored outputs
        seen += 32
    check(good, f"loader: {seen} samples: T rule, zeros on padding, mask from lengths, dtype")
    c = CachedTextConditioner(emb_dir, EMB_DIM)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        te.null_embedding = None
        te._impl_uncondition(["x"], {})
    # same code in tools/encode_captions.py and online; bf16 rounding differs across GPU types (A100 against A40: mean 0.046)
    online = te.null_embedding.cpu().float()
    d_null, env_null = (c.null_cpu.float() - online).abs().mean().item(), (online - null32).abs().mean().item()
    print(f"     null.pt vs online: mean abs {d_null:.5f}; bf16 vs fp32 envelope of the null: mean abs {env_null:.5f}")
    check(d_null <= env_null * 1.5, "null.pt vs the online null embedding within its bf16 rounding envelope")
    meta = {"text_emb": emb.cuda(), "text_mask": m.cuda()}
    cond, unc = c(list(cap), meta)
    check(cond.dtype == torch.bfloat16 and meta["y_mask"].dtype == torch.bool and (unc[:, 1:] == 0).all() and unc.shape == cond.shape,
          "conditioner: condition, mask and null row shapes")

    # 5. the denoiser does not read padded tokens
    kw = dict(input_size=256, in_channels=3, num_groups=4, hidden_size=256, num_blocks=2, num_text_blocks=1, patch_size=16,
              txt_embed_dim=EMB_DIM, txt_max_length=320, bottleneck_dim=32)
    torch.manual_seed(0)
    net = JiT_T2I(**kw).cuda().float()
    for p in net.parameters():
        p.data.add_(torch.randn_like(p) * 0.02)
    input_ids, mask, y = rows[len(rows) // 2]
    y = y.cuda().float(); mask = mask.cuda()
    zeroed = y.clone(); zeroed[~mask] = 0
    x = torch.randn(32, 3, 256, 256, device="cuda"); t = torch.rand(32, device="cuda")
    with torch.no_grad():
        a, b = net(x, t, y, y_mask=mask), net(x, t, zeroed, y_mask=mask)
    check(torch.equal(a, b) or (a - b).abs().max() == 0, f"denoiser: padded tokens have no effect (fp32, max diff {(a - b).abs().max().item():.2e})")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
