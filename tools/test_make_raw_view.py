#!/usr/bin/env python3
"""CPU test of tools/make_raw_view.py on a small runs root (RAW_ROOT). usage: python test_make_raw_view.py <scratch dir>
V1 link: a checkpoint whose EMA slot holds the raw weights' bytes, as independent tensors and as tensors on one storage,
   gets a symbolic link; a load through the link is the checkpoint
V2 copy: an EMA slot one bit apart, or a file that also holds optimizer state, gets a file: the EMA slot holds the raw
   weights, the raw weights are unchanged, the training-state keys are gone, raw_view_of names the checkpoint
V3 rerun: a second run changes nothing; with the checkpoint gone the tool fails and leaves the link, and the link is
   good again when the checkpoint is back; the checkpoints themselves never change (sha1 before and after); the config
   files are copied; no temporary file is left
exit code = number of failed tests"""
import hashlib, os, subprocess, sys, traceback
import torch

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_raw_view.py")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def sd_of(seed, ema_seed=None, share=False):
    g = torch.Generator().manual_seed(seed)
    raw = {f"blocks.{i}.w": torch.randn(32, 16 + i, generator=g) for i in range(4)}
    raw["b"] = torch.randn(8, generator=g).bfloat16()
    sd = {"denoiser." + k: v for k, v in raw.items()}
    g2 = torch.Generator().manual_seed(ema_seed) if ema_seed is not None else None
    for k, v in raw.items():
        sd["ema_denoiser." + k] = v if share else (torch.randn(v.shape, generator=g2).to(v.dtype) if g2 else v.clone())
    sd["conditioner.w"] = torch.randn(5, generator=g)
    return sd


def same(a, b):
    return a.keys() == b.keys() and all(a[k].dtype == b[k].dtype and torch.equal(a[k].float(), b[k].float()) for k in a)


def main():
    root = os.path.abspath(sys.argv[1])
    subprocess.run(["rm", "-rf", root]); os.makedirs(f"{root}/fam/run/exp_ladder_d64")
    run = f"{root}/fam/run/exp_ladder_d64"
    open(f"{run}/config-TrainerFn.FITTING-0000000000.yaml", "w").write("model: {}\n")
    bit = sd_of(3)
    bit["ema_denoiser.blocks.2.w"].view(torch.int32)[0, 0] ^= 1
    files = {100: {"global_step": 100, "state_dict": sd_of(1)},                       # equal bytes, two tensors
             200: {"global_step": 200, "state_dict": sd_of(2, share=True)},           # one storage
             300: {"global_step": 300, "state_dict": bit},                            # one bit apart
             400: {"global_step": 400, "state_dict": sd_of(4, ema_seed=9)},           # a moving EMA
             500: {"global_step": 500, "state_dict": sd_of(5), "optimizer_states": [{"m": torch.ones(3)}], "loops": {"a": 1},
                   "callbacks": {}, "lr_schedulers": []}}                             # equal bytes, but a training-state file
    for s, ck in files.items():
        torch.save(ck, f"{run}/ms-step={s:06d}.ckpt")
    before = {s: sha(f"{run}/ms-step={s:06d}.ckpt") for s in files}
    env = dict(os.environ, RAW_ROOT=root, RAW_SRC="fam/run", OMP_NUM_THREADS="2")
    call = lambda steps: subprocess.run([sys.executable, "-I", TOOL] + [a for s in steps for a in ("d64", str(s))], env=env, capture_output=True, text=True)
    r = call(files)
    view = lambda s: f"{root}/raw_view/fam/run/exp_ladder_d64/ms-step={s:06d}.ckpt"
    bad = 0

    def test(name, fn):
        nonlocal bad
        try:
            print(f"{name} PASS: {fn()}", flush=True)
        except Exception:
            bad += 1
            print(f"{name} FAIL:\n{traceback.format_exc()}\n{r.stdout}\n{r.stderr}", flush=True)

    def v1():
        assert r.returncode == 0, r.stderr
        for s in (100, 200):
            assert os.path.islink(view(s)) and os.path.realpath(view(s)) == os.path.realpath(f"{run}/ms-step={s:06d}.ckpt"), s
            got = torch.load(view(s), weights_only=False)
            assert got.keys() == files[s].keys() and got["global_step"] == s and same(got["state_dict"], files[s]["state_dict"]), s
        return "steps 100 and 200 are links; a load through the link is the checkpoint"

    def v2():
        for s in (300, 400, 500):
            assert os.path.isfile(view(s)) and not os.path.islink(view(s)), s
            got = torch.load(view(s), weights_only=False)
            src = files[s]["state_dict"]
            assert sorted(got) == ["global_step", "raw_view_of", "state_dict"], sorted(got)
            assert got["raw_view_of"] == f"{run}/ms-step={s:06d}.ckpt" and got["global_step"] == s
            want = dict(src)
            for k in src:
                if k.startswith("ema_denoiser."):
                    want[k] = src["denoiser." + k[len("ema_denoiser."):]]
            assert same(got["state_dict"], want), s
            a, b = got["state_dict"]["ema_denoiser.blocks.2.w"], got["state_dict"]["denoiser.blocks.2.w"]
            assert torch.equal(a.view(torch.int32), b.view(torch.int32))
        return "steps 300 (one bit apart), 400 (moving EMA) and 500 (optimizer state) are files with the raw weights in the EMA slot"

    def v3():
        nonlocal r
        m = {s: os.lstat(view(s)).st_mtime_ns for s in files}
        r = call(files)
        assert r.returncode == 0 and r.stdout.count("exists") == 5, r.stdout
        assert m == {s: os.lstat(view(s)).st_mtime_ns for s in files}
        os.rename(f"{run}/ms-step=000100.ckpt", f"{run}/gone")
        assert not os.path.exists(view(100)) and os.path.islink(view(100))
        r = call([100])
        assert r.returncode != 0 and os.path.islink(view(100)), "a missing checkpoint must fail"
        os.rename(f"{run}/gone", f"{run}/ms-step=000100.ckpt")
        r = call([100])
        assert r.returncode == 0 and "exists" in r.stdout and os.path.exists(view(100)), r.stdout + r.stderr
        assert same(torch.load(view(100), weights_only=False)["state_dict"], files[100]["state_dict"])
        assert before == {s: sha(f"{run}/ms-step={s:06d}.ckpt") for s in files}, "a checkpoint changed"
        assert os.path.isfile(f"{root}/raw_view/fam/run/exp_ladder_d64/config-TrainerFn.FITTING-0000000000.yaml")
        left = [f for f in os.listdir(os.path.dirname(view(100))) if ".tmp" in f]
        assert not left, left
        return "a second run writes nothing; a missing checkpoint fails and its link is good again when it is back; the 5 checkpoints are unchanged; config copied"

    test("V1", v1); test("V2", v2); test("V3", v3)
    print("ok" if not bad else f"FAIL ({bad})")
    sys.exit(bad)


if __name__ == "__main__":
    main()
