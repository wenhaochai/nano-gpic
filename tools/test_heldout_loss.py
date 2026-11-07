#!/usr/bin/env python3
"""CPU tests of tools/heldout_loss.py on a tiny JiT_T2I.
Run on a vis node from baselines/PixelGen:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONPATH=. python ../../tools/test_heldout_loss.py [H1 H2 ...]
H1 loss: evaluate() equals JiTTrainer's own loss (its __call__, with t and the noise patched in) on the same images,
   captions (masked, padded per batch), t and noise, per image and per level, to fp32 rounding; the batch size does not
   change the result beyond rounding
H2 levels: the 16 t equal sigmoid(P_mean + P_std Phi^-1((k + 0.5) / 16)) then the time shift, from an independent formula
   (statistics.NormalDist), for two time shifts; the noise of image j is FixedValLoss's draw
H3 repeat: two evaluations give identical numbers (bit for bit)
H4 input: to_input equals val_transform's ToTensor + Normalize bit for bit, and text_batch equals the cached loader's
   batch rule (T = longest rounded up to 8, at most 320; zeros and False after each caption)
H5 tasks: an end-to-end CPU run on fake ms checkpoints (strict load of the EMA and the raw weights from a run config)
   writes one line per (checkpoint, weights) with the asked fields; the task list skips done entries, lists a
   checkpoint with one weights missing, and a second run adds no line; the lockf counter hands out each index once
H6 train shard: prep --train-shard (its CPU path, with fake shards and a fake text encoder) copies the val part of the
   default file bit for bit and reads the train part from the given shard and its cache; the data file names carry the
   shard (00787's name unchanged); the val loss of one fake checkpoint is the same with both files, the lines record
   the train shard and are keyed by the data file; with REAL=1, also the two real 3072 files (val tensors bit-identical)
Each test prints PASS or FAIL; the exit code is the number of failed tests."""
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import traceback
from contextlib import contextmanager, redirect_stdout
import io

import numpy as np
import torch

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import heldout_loss as H   # noqa: E402

from src.diffusion.flow_matching import training_JiT   # noqa: E402
from src.models.transformer.JiT_T2I import JiT_T2I   # noqa: E402

torch.set_default_dtype(torch.float32)
SMALL = dict(input_size=64, in_channels=3, num_groups=4, hidden_size=64, num_blocks=2, num_text_blocks=1, patch_size=16,
             txt_embed_dim=32, txt_max_length=32, bottleneck_dim=16)
LOSS = dict(timeshift=1.0, P_mean=-0.8, P_std=0.8, t_eps=0.05)
RESULTS = {}


def fake_split(n, seed=0, img=64, dim=32, max_len=20):
    g = torch.Generator().manual_seed(seed)
    imgs = torch.randint(0, 256, (n, img, img, 3), generator=g, dtype=torch.uint8)
    lens = torch.randint(1, max_len + 1, (n,), generator=g).tolist()
    embs = [torch.randn(L, dim, generator=g) for L in lens]
    s = H.pack_split(None, imgs, embs, {})
    s["text_max_length"] = SMALL["txt_max_length"]
    s["lens"], s["offs"] = s["lens"].tolist(), s["offs"].tolist()
    s["order"] = sorted(range(n), key=lambda i: (s["lens"][i], i))
    return s


def tiny_net(seed):
    torch.manual_seed(seed)
    net = JiT_T2I(**SMALL).eval()
    with torch.no_grad():   # FinalLayer starts at zero; give every weight a value so the test sees the whole network
        for p in net.parameters():
            if p.requires_grad:
                p.add_(0.05 * torch.randn_like(p))
    return net


@contextmanager
def patched_trainer(t_value, noise):
    """JiTTrainer draws t = time_shift_fn(base_t) and noise = randn_like(x); patch both to the given values."""
    orig_shift, orig_rl = training_JiT.time_shift_fn, torch.randn_like
    training_JiT.time_shift_fn = lambda base_t, shift: t_value.expand(base_t.shape[0]).clone()
    torch.randn_like = lambda x, *a, **k: noise.clone()
    try:
        yield
    finally:
        training_JiT.time_shift_fn, torch.randn_like = orig_shift, orig_rl


def test(fn):
    def run():
        try:
            fn()
            RESULTS[fn.__name__] = True
            print(f"PASS {fn.__name__}: {fn.__doc__.splitlines()[0]}", flush=True)
        except Exception:
            RESULTS[fn.__name__] = False
            print(f"FAIL {fn.__name__}: {fn.__doc__.splitlines()[0]}\n{traceback.format_exc()}", flush=True)
    run.__name__ = fn.__name__
    return run


@test
def H1():
    """evaluate() equals JiTTrainer's own loss for the same images, captions, t and noise"""
    split = fake_split(7)
    nets = {"a": tiny_net(1), "b": tiny_net(2)}
    tr = H.make_trainer(LOSS)
    ts = H.levels(LOSS["P_mean"], LOSS["P_std"], LOSS["timeshift"])
    res = H.evaluate(nets, tr, split, ts, seed=3, batch=3, threads=2)
    N, num_t = 7, len(ts)
    shape = (3, 64, 64)
    for name, net in nets.items():
        per = torch.zeros(num_t, N, dtype=torch.float64)
        # JiTTrainer on batches in a different grouping (images 0..6 in index order, batches of 4 and 3)
        for idx in ([0, 1, 2, 3], [4, 5, 6]):
            it = torch.tensor(idx)
            x = H.to_input(split["img"][it])
            y, mask = H.text_batch(split, idx, "cpu")
            for k in range(num_t):
                noise = torch.stack([H.image_noise(3, j, num_t, shape)[k] for j in idx])
                with patched_trainer(ts[k], noise):
                    out = tr(net, None, None, x, y, torch.zeros_like(y), {"y_mask": mask.clone()})
                per[k, it] = out["loss_per_sample"].double()
        ref_level, ref_image = per.mean(1), per.mean(0)
        got_level, got_image = res[name]["level_mean"], res[name]["image_mean"]
        rel = ((got_level - ref_level).abs() / ref_level.abs()).max().item()
        rel_i = ((got_image - ref_image).abs() / ref_image.abs()).max().item()
        print(f"  net {name}: loss {got_level.mean():.6f} vs JiTTrainer {ref_level.mean():.6f}; max rel diff per level "
              f"{rel:.2e}, per image {rel_i:.2e}")
        assert rel < 2e-5 and rel_i < 2e-5, (rel, rel_i)
        assert got_level.std() > 0 and ref_level.mean() > 0
    # the batch size changes only rounding
    res1 = H.evaluate(nets, tr, split, ts, seed=3, batch=7, threads=2)
    d = max(((res1[k]["level_mean"] - res[k]["level_mean"]).abs() / res[k]["level_mean"]).max().item() for k in nets)
    print(f"  batch 7 vs batch 3: max rel diff {d:.2e}")
    assert d < 2e-5, d
    # another seed changes the numbers
    res2 = H.evaluate(nets, tr, split, ts, seed=4, batch=3, threads=2)
    assert not torch.equal(res2["a"]["level_mean"], res["a"]["level_mean"])


@test
def H2():
    """the levels match the formula; the noise is FixedValLoss's draw"""
    nd = statistics.NormalDist()
    for shift in (1.0, 3.0):
        for P_mean, P_std in ((-0.8, 0.8), (0.3, 1.2)):
            got = H.levels(P_mean, P_std, shift)
            assert got.dtype == torch.float32 and got.shape == (16,)
            for k in range(16):
                z = nd.inv_cdf((k + 0.5) / 16)
                b = 1 / (1 + math.exp(-(P_mean + P_std * z)))
                want = b / (b + (1 - b) * shift)
                assert abs(got[k].item() - want) < 1e-6, (shift, P_mean, P_std, k, got[k].item(), want)
            assert torch.all(got[1:] > got[:-1])
    ts = H.levels(-0.8, 0.8, 1.0)
    print(f"  t (P_mean -0.8, P_std 0.8, shift 1): {[round(v, 4) for v in ts.tolist()]}")
    # FixedValLoss: g = torch.Generator().manual_seed(seed * 1_000_003 + j); torch.randn((num_t, *x.shape[1:]), generator=g)
    for seed, j in ((0, 0), (0, 4095), (2, 17)):
        g = torch.Generator().manual_seed(seed * 1_000_003 + j)
        want = torch.randn((16, 3, 8, 8), generator=g)
        assert torch.equal(H.image_noise(seed, j, 16, (3, 8, 8)), want)
        buf = torch.empty(16, 3, 8, 8)
        H.image_noise(seed, j, 16, (3, 8, 8), out=buf)
        assert torch.equal(buf, want)


@test
def H3():
    """two evaluations give identical numbers"""
    split = fake_split(5, seed=1)
    nets = {"a": tiny_net(3)}
    tr = H.make_trainer(LOSS)
    ts = H.levels(-0.8, 0.8, 1.0)
    r1 = H.evaluate(nets, tr, split, ts, seed=0, batch=2, threads=1)
    r2 = H.evaluate(nets, tr, split, ts, seed=0, batch=2, threads=3)
    assert torch.equal(r1["a"]["level_mean"], r2["a"]["level_mean"]) and torch.equal(r1["a"]["image_mean"], r2["a"]["image_mean"])


@test
def H4():
    """to_input equals val_transform's tensor steps; text_batch follows the cached loader's rule"""
    from PIL import Image
    from src.data.dataset.gpic_dataset import val_transform
    g = torch.Generator().manual_seed(0)
    u8 = torch.randint(0, 256, (2, 256, 256, 3), generator=g, dtype=torch.uint8)
    post = val_transform.transforms[1:]
    for i in range(2):
        pil = Image.fromarray(u8[i].numpy())
        want = post[1](post[0](pil))
        assert torch.equal(H.to_input(u8[i:i + 1])[0], want)
    # the full transform on a non-square image equals center_crop_u8 + to_input
    big = Image.fromarray(np.random.default_rng(0).integers(0, 256, (300, 520, 3), dtype=np.uint8))
    b = io.BytesIO(); big.save(b, format="PNG")
    want = val_transform(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))
    got = H.to_input(torch.from_numpy(H.center_crop_u8(b.getvalue()))[None])[0]
    assert torch.equal(got, want)
    # text batch
    embs = [torch.randn(L, 4) for L in (3, 9, 1)]
    s = H.pack_split(None, torch.zeros(3, 2, 2, 3, dtype=torch.uint8), embs, {})
    s["lens"], s["offs"] = s["lens"].tolist(), s["offs"].tolist()
    y, m = H.text_batch(s, [0, 1, 2], "cpu")
    assert y.shape == (3, 16, 4) and m.shape == (3, 16)   # longest 9 -> 16
    for r, e in enumerate(embs):
        assert torch.equal(y[r, :len(e)], e) and not y[r, len(e):].any()
        assert m[r, :len(e)].all() and not m[r, len(e):].any()
    y, m = H.text_batch(s, [0, 2], "cpu")
    assert y.shape == (2, 8, 4)
    s["lens"][1] = 318; s["text_max_length"] = 320
    s["emb"] = torch.cat([embs[0], torch.randn(318, 4), embs[2]]); s["offs"] = [0, 3, 321]
    assert H.text_batch(s, [1], "cpu")[0].shape == (1, 320, 4)   # 318 -> 320, the cap


def write_run(root, size, steps, seeds):
    run = os.path.join(root, f"exp_ladder_{size}")
    os.makedirs(run)
    init = dict(SMALL, weight_path=None, load_ema=False)
    cfg = {"model": {"denoiser": {"class_path": "src.models.transformer.JiT_T2I.JiT_T2I", "init_args": init},
                     "diffusion_trainer": {"class_path": "src.diffusion.flow_matching.training_JiT.JiTTrainer", "init_args": {
                         "scheduler": {"class_path": "src.diffusion.flow_matching.scheduling.LinearScheduler"},
                         "loss_weight_fn": "src.diffusion.flow_matching.training_JiT.constant", "lognorm_t": True,
                         "timeshift": 1.0, "P_mean": -0.8, "P_std": 0.8, "t_eps": 0.05, "null_condition_p": 0.0}}}}
    import yaml
    with open(os.path.join(run, "config-TrainerFn.FITTING-0000000000.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    paths = []
    for step, (sr, se) in zip(steps, seeds):
        sd = {}
        sd.update({"denoiser." + k: v for k, v in tiny_net(sr).state_dict().items()})
        sd.update({"ema_denoiser." + k: v for k, v in tiny_net(se).state_dict().items()})
        p = os.path.join(run, f"ms-step={step:06d}.ckpt")
        torch.save({"global_step": step, "state_dict": sd}, p)
        os.utime(p, (1, 1))   # old, so --min-age passes
        paths.append(p)
    return paths


def run_main(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = H.main(argv)
    return rc, buf.getvalue()


@test
def H5():
    """end-to-end CPU run on fake checkpoints; the task list skips done entries; the counter"""
    with tempfile.TemporaryDirectory() as tmp:
        ck = write_run(tmp, "d64", [100, 200], [(11, 12), (13, 14)])
        n_val, n_train = 4, 3
        data = dict(version=1, n_val=n_val, n_train=n_train, train_shard="/x/gpic_train_00787.tar",
                    val=H.pack_split(None, fake_split(n_val, 5)["img"], [torch.randn(L, 32) for L in (3, 5, 2, 7)], {}),
                    train=H.pack_split(None, fake_split(n_train, 6)["img"], [torch.randn(L, 32) for L in (4, 1, 6)], {}))
        for s in ("val", "train"):
            data[s]["text_max_length"] = SMALL["txt_max_length"]
        dpath = os.path.join(tmp, "data.pt")
        torch.save(data, dpath)
        out = os.path.join(tmp, "res.jsonl")
        common = ["--n-val", str(n_val), "--n-train", str(n_train), "--data", dpath]
        # the list: both checkpoints, longest first is a tie; a checkpoint younger than --min-age is left out
        rc, txt = run_main(["list", "--out", out, "--min-age", "600", "--ckpt", *ck] + common)
        assert len([l for l in txt.splitlines() if l.startswith("{")]) == 2, txt
        os.utime(ck[1], None)
        rc, txt = run_main(["list", "--out", out, "--min-age", "600", "--ckpt", *ck] + common)
        assert len([l for l in txt.splitlines() if l.startswith("{")]) == 1, txt
        os.utime(ck[1], (1, 1))
        # the run: one checkpoint by --ckpt, on CPU
        rc, txt = run_main(["work", "--out", out, "--device", "cpu", "--batch", "3", "--threads", "1",
                            "--ckpt", ck[0]] + common)
        assert rc == 0, txt
        recs = [json.loads(l) for l in open(out)]
        assert [r["weights"] for r in recs] == ["ema", "raw"], recs
        need = ("run_root", "ladder", "size", "d", "nb", "nt", "step", "weights", "n_val", "n_train", "num_t", "seed",
                "loss_val", "loss_train", "loss_val_t", "loss_train_t", "t", "elapsed_s", "ckpt", "data")
        for r in recs:
            assert all(k in r for k in need), [k for k in need if k not in r]
            assert (r["size"], r["d"], r["nb"], r["nt"], r["step"], r["num_t"]) == ("d64", 64, 2, 1, 100, 16)
            assert len(r["loss_val_t"]) == 16 and abs(sum(r["loss_val_t"]) / 16 - r["loss_val"]) < 1e-9
            assert r["run_root"] == tmp and r["ladder"] == "lite"
        assert recs[0]["loss_val"] != recs[1]["loss_val"]   # EMA and raw weights differ
        # the same numbers as evaluate() on nets built directly
        tr = H.make_trainer(LOSS)
        ts = H.levels(-0.8, 0.8, 1.0)
        d2 = H.load_data(dpath, "cpu")
        direct = H.evaluate({"ema": tiny_net(12), "raw": tiny_net(11)}, tr, d2["val"], ts, seed=0, batch=3, threads=1)
        for r in recs:
            assert abs(float(direct[r["weights"]]["level_mean"].mean()) - r["loss_val"]) < 1e-6 * abs(r["loss_val"]), r["weights"]
        # the list now skips ckpt 0; with its raw line removed, it lists ckpt 0 again (one weights missing)
        rc, txt = run_main(["list", "--out", out, "--min-age", "600", "--ckpt", *ck] + common)
        tasks = [json.loads(l) for l in txt.splitlines() if l.startswith("{")]
        assert [t["step"] for t in tasks] == [200], tasks
        lines = open(out).read().splitlines()
        with open(out, "w") as f:
            f.write(lines[0] + "\n")
        rc, txt = run_main(["list", "--out", out, "--min-age", "600", "--ckpt", *ck] + common)
        assert sorted(json.loads(l)["step"] for l in txt.splitlines() if l.startswith("{")) == [100, 200]
        # other N: nothing is done
        rc, txt = run_main(["list", "--out", out, "--min-age", "600", "--ckpt", *ck, "--n-val", "5", "--n-train", "3", "--data", dpath[:-3] + "_other.pt"])
        assert len([l for l in txt.splitlines() if l.startswith("{")]) == 2
        # the task-list path: two workers in turn on one counter; the raw of ckpt 0 and both weights of ckpt 1 get done
        tpath, cpath = os.path.join(tmp, "tasks.jsonl"), os.path.join(tmp, "counter")
        with open(tpath, "w") as f:
            f.write("\n".join(json.dumps(H.spec_of(c)) for c in ck) + "\n")
        for _ in range(2):
            rc, txt = run_main(["work", "--out", out, "--device", "cpu", "--batch", "3", "--threads", "1",
                                "--tasks", tpath, "--counter", cpath] + common)
            assert rc == 0, txt
        recs = [json.loads(l) for l in open(out)]
        assert sorted((r["step"], r["weights"]) for r in recs) == [(100, "ema"), (100, "raw"), (200, "ema"), (200, "raw")], recs
        assert recs[1]["weights_in_task"] == ["raw"]   # only the missing weights were computed
        r100 = {r["weights"]: r for r in recs if r["step"] == 100}
        assert abs(r100["raw"]["loss_val"] - float(direct["raw"]["level_mean"].mean())) < 1e-6 * r100["raw"]["loss_val"]
        # a second run adds nothing
        os.remove(cpath)
        rc, txt = run_main(["work", "--out", out, "--device", "cpu", "--tasks", tpath, "--counter", cpath] + common)
        assert rc == 0 and len(open(out).read().splitlines()) == 4
        rc, txt = run_main(["list", "--out", out, "--min-age", "600", "--ckpt", *ck] + common)
        assert not [l for l in txt.splitlines() if l.startswith("{")]
        # a wrong-size checkpoint fails loudly (strict load), and the worker reports the failure
        bad = write_run(os.path.join(tmp, "bad"), "d64", [300], [(1, 2)])[0]
        sd = torch.load(bad, weights_only=False)
        sd["state_dict"].pop("ema_denoiser.final_layer.linear.weight")
        torch.save(sd, bad)
        rc, txt = run_main(["work", "--out", out, "--device", "cpu", "--ckpt", bad] + common)
        assert rc == 1 and "FAILED" in txt and len(open(out).read().splitlines()) == 4
        # the counter: 4 processes x 25 claims give 0..99 once each
        cpath2 = os.path.join(tmp, "counter2")
        code = (f"import sys; sys.path.insert(0, {TOOLS!r}); import heldout_loss as H; "
                f"print(' '.join(str(H.claim({cpath2!r})) for _ in range(25)))")
        env = dict(os.environ, PYTHONPATH=os.getcwd())
        procs = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env=env) for _ in range(4)]
        got = sorted(int(v) for p in procs for v in p.communicate()[0].split())
        assert got == list(range(100)), got


def write_tar(path, samples):
    """A webdataset tar: <key>.jpg and <key>.json (caption, caption_type) per sample."""
    import tarfile
    from PIL import Image
    with tarfile.open(path, "w") as tar:
        for key, arr, cap in samples:
            for ext, b in (("jpg", None), ("json", json.dumps(dict(caption=cap, caption_type="x")).encode())):
                if b is None:
                    bio = io.BytesIO(); Image.fromarray(arr).save(bio, format="JPEG"); b = bio.getvalue()
                ti = tarfile.TarInfo(f"{key}.{ext}"); ti.size = len(b)
                tar.addfile(ti, io.BytesIO(b))


def write_cache(emb_dir, name, keys, lens, seed):
    """A training text cache for one shard: <name>.emb (bf16 rows of EMB_DIM) and <name>.idx.npz (keys, offsets, lengths)."""
    g = torch.Generator().manual_seed(seed)
    embs = [torch.randn(L, H.EMB_DIM, generator=g).bfloat16() for L in lens]
    offs = np.cumsum([0] + list(lens[:-1]))
    with open(os.path.join(emb_dir, name + ".emb"), "wb") as f:
        f.write(torch.cat(embs).view(torch.int16).numpy().tobytes())
    np.savez(os.path.join(emb_dir, name + ".idx.npz"), keys=np.array(keys), offsets=offs, lengths=np.array(lens))
    return embs


class FakeEncoder(torch.nn.Module):
    """Stands in for Qwen3TextEncoder in prep: the tokenizer gives one token per word; _impl_condition returns the cache's
    rows (CACHE[caption]) plus 1e-3 noise, so the text check passes and the re-encode differs from the cache."""
    CACHE = {}

    def __init__(self, *a, **k):
        super().__init__()

        class Tok:
            eos_token_id, pad_token_id = 0, 0

            def __call__(self, caps, truncation, max_length):   # captions are unique, so are their ids
                return {"input_ids": [[(hash(c) + i) % 1000 + 1 for i in range(len(c.split()))][:max_length] for c in caps]}
        self.tokenizer = Tok()
        self.model = torch.nn.Identity()   # prep's CPU path turns its compile off

    def _impl_condition(self, y, meta):
        ids, mask = meta["input_ids"], meta["text_mask"]
        out = torch.zeros(ids.shape[0], ids.shape[1], H.EMB_DIM, dtype=torch.bfloat16)
        for r in range(ids.shape[0]):
            key = tuple(ids[r][mask[r]].tolist())
            e = FakeEncoder.CACHE[key]
            out[r, :len(e)] = (e.float() + 1e-3 * torch.randn_like(e.float())).bfloat16()
        return out


@test
def H6():
    """--train-shard: the val part is copied bit for bit; the val loss is the same with both files"""
    import types
    assert os.path.basename(H.data_path(3072, 3072)) == "data_v1_val3072_train3072_00787.pt"
    assert os.path.basename(H.data_path(3072, 3072, "gpic_train_00000")) == "data_v1_val3072_train3072_00000.pt"
    assert H.train_shard_path("gpic_train_00000.tar") == H.train_shard_path("/x/gpic_train_00000") == \
        os.path.join(H.TRAIN_DIR, "gpic_train_00000.tar")
    rng = np.random.default_rng(0)
    words = "a red cat sits on the mat under blue sky with two dogs near old tree".split()
    with tempfile.TemporaryDirectory() as tmp:
        tdir, vdir, edir, odir = (os.path.join(tmp, x) for x in ("train", "val", "emb", "out"))
        for d in (tdir, vdir, edir):
            os.makedirs(d)
        n = 5

        def samples(prefix, k):
            return [(f"{prefix}{i:04d}", rng.integers(0, 256, (40 + 8 * i, 72, 3), dtype=np.uint8),
                     " ".join(list(rng.choice(words, int(rng.integers(1, 9)))) + [f"{prefix}{i}"])) for i in range(k)]
        val_s = samples("v", n)
        write_tar(os.path.join(vdir, "gpic_val_00000.tar"), val_s)
        enc = FakeEncoder()
        for shard, seed in (("gpic_train_00787", 1), ("gpic_train_00000", 2)):
            tr_s = samples(shard[-5:], n + 2)
            write_tar(os.path.join(tdir, shard + ".tar"), tr_s)
            ids = enc.tokenizer([c for _, _, c in tr_s], True, 320)["input_ids"]
            embs = write_cache(edir, shard, [k for k, _, _ in tr_s], [len(i) for i in ids], seed)
            FakeEncoder.CACHE.update({tuple(i): e for i, e in zip(ids, embs)})
        ids = enc.tokenizer([c for _, _, c in val_s], True, 320)["input_ids"]
        FakeEncoder.CACHE.update({tuple(i): torch.randn(len(i), H.EMB_DIM).bfloat16() for i in ids})
        # prep with the fake shards, cache and encoder (CPU path)
        saved = (H.TRAIN_DIR, H.TRAIN_EMB, H.VAL_GLOB, H.OUT_DIR, H.IMAGE)
        fake_mod = types.ModuleType("fake_qwen"); fake_mod.Qwen3TextEncoder = FakeEncoder
        real_mod = sys.modules.get("src.models.conditioner.qwen3_text_encoder")
        H.TRAIN_DIR, H.TRAIN_EMB, H.VAL_GLOB, H.OUT_DIR, H.IMAGE = tdir, edir, os.path.join(vdir, "gpic_val_*.tar"), odir, 32
        sys.modules["src.models.conditioner.qwen3_text_encoder"] = fake_mod
        try:
            common = ["--n-val", str(n), "--n-train", str(n), "--procs", "2", "--device", "cpu"]
            rc, txt = run_main(["prep"] + common)
            p787 = os.path.join(odir, f"data_v1_val{n}_train{n}_00787.pt")
            assert os.path.exists(p787), txt
            # the val copy keeps the source's encode: change the fake encoder's val rows, so a re-encode would differ
            for i in ids:
                FakeEncoder.CACHE[tuple(i)] = torch.randn(len(i), H.EMB_DIM).bfloat16()
            rc, txt = run_main(["prep", "--train-shard", "gpic_train_00000"] + common)
            p000 = os.path.join(odir, f"data_v1_val{n}_train{n}_00000.pt")
            assert os.path.exists(p000) and "bit for bit" in txt, txt
        finally:
            H.TRAIN_DIR, H.TRAIN_EMB, H.VAL_GLOB, H.OUT_DIR, H.IMAGE = saved
            if real_mod is not None:
                sys.modules["src.models.conditioner.qwen3_text_encoder"] = real_mod
            else:
                sys.modules.pop("src.models.conditioner.qwen3_text_encoder", None)
        a, b = (torch.load(p, weights_only=False) for p in (p787, p000))
        assert os.path.basename(a["train_shard"]) == "gpic_train_00787.tar" and os.path.basename(b["train_shard"]) == "gpic_train_00000.tar"
        assert b["val_from"] == p787 and a["val_from"] is None
        for k, v in a["val"].items():
            assert (torch.equal(v, b["val"][k]) and v.dtype == b["val"][k].dtype) if torch.is_tensor(v) else v == b["val"][k], k
        assert not torch.equal(a["train"]["img"], b["train"]["img"])
        assert all(s.startswith("00000") for s in b["train"]["keys"]) and b["train"]["shards"][0] == "gpic_train_00000.tar"
        # the train part is the 00000 cache's rows, in sample order
        name = "gpic_train_00000"
        with np.load(os.path.join(edir, name + ".idx.npz")) as z:
            raw = np.fromfile(os.path.join(edir, name + ".emb"), dtype=np.int16)
            rows = torch.from_numpy(raw).view(torch.bfloat16).view(-1, H.EMB_DIM)
            want = torch.cat([rows[o:o + L] for o, L in zip(z["offsets"][:n], z["lengths"][:n])])
        assert torch.equal(b["train"]["emb"], want)
        assert 0 < b["text_check"]["rel_rms"] < 3e-2, b["text_check"]
        print(f"  prep: val part bit-identical ({len(a['val'])} fields); train from {b['train']['shards'][0]}; "
              f"text check rel_rms {b['text_check']['rel_rms']:.2e}")
        # one fake checkpoint on both files: same val loss, different train loss, two lines per file
        ck = write_run(tmp, "d64", [100], [(21, 22)])
        out = os.path.join(tmp, "res.jsonl")
        for p in (p787, p000):
            d = torch.load(p, weights_only=False)
            for s_ in ("val", "train", "train_reenc"):   # the tiny net's caption width and image size
                d[s_]["emb"] = d[s_]["emb"][:, :SMALL["txt_embed_dim"]].float().contiguous()
                d[s_]["img"] = torch.nn.functional.interpolate(d[s_]["img"].permute(0, 3, 1, 2).float(), size=64).round()\
                    .clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
                d[s_]["text_max_length"] = SMALL["txt_max_length"]
            torch.save(d, p)
        for p, shard in ((p787, "gpic_train_00787"), (p000, "gpic_train_00000")):
            rc, txt = run_main(["work", "--out", out, "--device", "cpu", "--batch", "3", "--threads", "1", "--n-val", str(n),
                                "--n-train", str(n), "--data", p, "--train-shard", shard, "--ckpt", ck[0]])
            assert rc == 0, txt
            # the list keys on the data file: done for this file only
            rc, txt = run_main(["list", "--out", out, "--min-age", "0", "--n-val", str(n), "--n-train", str(n), "--data", p,
                                "--ckpt", ck[0]])
            assert not [l for l in txt.splitlines() if l.startswith("{")], txt
        recs = [json.loads(l) for l in open(out)]
        assert [(r["train_shard"], r["weights"]) for r in recs] == [("gpic_train_00787.tar", "ema"), ("gpic_train_00787.tar", "raw"),
                                                                    ("gpic_train_00000.tar", "ema"), ("gpic_train_00000.tar", "raw")]
        assert [r["data"] for r in recs] == [os.path.basename(p787)[:-3]] * 2 + [os.path.basename(p000)[:-3]] * 2
        for w in ("ema", "raw"):
            r1, r2 = (r for r in recs if r["weights"] == w)
            assert r1["loss_val"] == r2["loss_val"] and r1["loss_val_t"] == r2["loss_val_t"], (r1["loss_val"], r2["loss_val"])
            assert r1["loss_train"] != r2["loss_train"]
            print(f"  {w}: loss_val {r1['loss_val']:.6f} on both files; loss_train {r1['loss_train']:.6f} (00787) "
                  f"{r2['loss_train']:.6f} (00000)")
        # a wrong --train-shard for the file fails
        try:
            run_main(["work", "--out", out, "--device", "cpu", "--n-val", str(n), "--n-train", str(n), "--data", p000,
                      "--train-shard", "gpic_train_00787", "--ckpt", ck[0]])
            raise RuntimeError("no assert")
        except AssertionError:
            pass
    if os.environ.get("REAL") == "1":   # the two real files
        a, b = (torch.load(H.data_path(3072, 3072, s), map_location="cpu", mmap=True, weights_only=False)
                for s in ("gpic_train_00787", "gpic_train_00000"))
        for k, v in a["val"].items():
            assert (torch.equal(v, b["val"][k]) and v.dtype == b["val"][k].dtype) if torch.is_tensor(v) else v == b["val"][k], k
        print(f"  real files: val part bit-identical ({a['val']['img'].shape[0]} images, {a['val']['emb'].shape[0]} tokens); "
              f"train {os.path.basename(a['train_shard'])} vs {os.path.basename(b['train_shard'])}")


def main():
    tests = [H1, H2, H3, H4, H5, H6]
    want = set(sys.argv[1:])
    for t in tests:
        if not want or t.__name__ in want:
            t()
    fails = sum(not v for v in RESULTS.values())
    print(f"{len(RESULTS) - fails}/{len(RESULTS)} passed", flush=True)
    sys.exit(fails)


if __name__ == "__main__":
    main()
