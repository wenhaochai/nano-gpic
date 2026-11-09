#!/usr/bin/env python3
"""Raw-weight view of ladder checkpoints. The evaluation samples with ema_denoiser.*, so a checkpoint whose ema_denoiser.*
holds the raw weights (denoiser.*) evaluates the raw model with the unchanged evaluation code. For each <size> <step>:
runs/gpic-speed/raw_view/<rel>/exp_ladder_<size>/ms-step=<step>.ckpt beside copies of the run's config files, so
jobs/evals_gmflow.sh runs on the root runs/gpic-speed/raw_view/<rel>. (From tmp/ladder_prof/make_raw_view.py, 2026-11-07.)
2026-11-09: a no-EMA checkpoint already holds the raw weights in its EMA slot, so its view was a second copy of the file
(188 views, 0.72 TiB). When every ema_denoiser.X has the bytes of denoiser.X (same keys; one storage, or compared bit for
bit) the view is a symbolic link to the checkpoint; when not, it is the copy as before (state_dict with the EMA slot
replaced, the training-state keys dropped, raw_view_of added).
usage: RAW_SRC=<rel> make_raw_view.py <size> <step> [<size> <step> ...]
RAW_SRC: the run root under runs/gpic-speed (default muon_full); RAW_ROOT: another runs/gpic-speed (the tests)."""
import glob, os, shutil, sys
import torch

_R = os.environ.get("RAW_ROOT", "/scratch/gpfs/GROUP/USER/runs/gpic-speed").rstrip("/") + "/"
_rel = os.environ.get("RAW_SRC", "muon_full").strip("/")
SRC = _R + _rel
DST = _R + "raw_view/" + _rel
_BITS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}   # element size -> a dtype to compare bytes with


def twin_weights(sd):
    """True when every ema_denoiser.X holds the bytes of denoiser.X, with the same keys on both sides."""
    ema = {k[len("ema_denoiser."):]: v for k, v in sd.items() if k.startswith("ema_denoiser.")}
    raw = {k[len("denoiser."):]: v for k, v in sd.items() if k.startswith("denoiser.")}
    if not ema or ema.keys() != raw.keys():
        return False
    for k, e in ema.items():
        r = raw[k]
        if (e.dtype, e.shape) != (r.dtype, r.shape) or e.element_size() not in _BITS:
            return False
        if e.numel() == 0 or (e.data_ptr() == r.data_ptr() and e.stride() == r.stride()):
            continue
        i = _BITS[e.element_size()]
        if not torch.equal(e.contiguous().view(i), r.contiguous().view(i)):
            return False
    return True


def main(args):
    assert args and len(args) % 2 == 0, __doc__
    for size, step in zip(args[0::2], args[1::2]):
        src = f"{SRC}/exp_ladder_{size}/ms-step={int(step):06d}.ckpt"
        out_dir = f"{DST}/exp_ladder_{size}"; os.makedirs(out_dir, exist_ok=True)
        out = f"{out_dir}/ms-step={int(step):06d}.ckpt"
        if os.path.exists(out):
            print("exists", out); continue
        ck = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
        sd = ck["state_dict"]
        raw = {k[len("denoiser."):]: v for k, v in sd.items() if k.startswith("denoiser.")}
        ema = [k for k in sd if k.startswith("ema_denoiser.")]
        assert len(raw) == len(ema) and all(k[len("ema_denoiser."):] in raw for k in ema), "denoiser and ema_denoiser differ in keys"
        tmp = out + f".tmp{os.getpid()}"
        # a link only to a file with nothing but the keys a view keeps (an ms checkpoint; never a last.ckpt with its optimizer)
        if twin_weights(sd) and not any(k in ck for k in ("optimizer_states", "lr_schedulers", "callbacks", "loops")):
            os.symlink(src, tmp); os.replace(tmp, out)
            what = f"linked {out} -> {src}"
        else:
            new = dict(sd)
            for k in ema:
                new[k] = raw[k[len("ema_denoiser."):]].clone()
            keep = {k: v for k, v in ck.items() if k not in ("optimizer_states", "lr_schedulers", "callbacks", "loops")}
            keep["state_dict"] = new
            keep["raw_view_of"] = src
            torch.save(keep, tmp); os.replace(tmp, out)
            what = f"wrote {out} {os.path.getsize(out) / 2**30:.2f} GiB"
        for c in glob.glob(f"{SRC}/exp_ladder_{size}/config-*.yaml"):
            dst = os.path.join(out_dir, os.path.basename(c))
            if not os.path.exists(dst):
                shutil.copy(c, dst)
        print(what)


if __name__ == "__main__":
    main(sys.argv[1:])
