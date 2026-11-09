#!/usr/bin/env python3
"""CPU test of the twins rule of src/plugins/async_ckpt_io.AsyncShmCheckpointIO (2026-11-09): a state_dict tensor
ema_denoiser.X with the bytes of denoiser.X is written once.
T1 content: a checkpoint with equal pairs (fp32, bf16, one with a NaN), a pair that differs by one sign bit (-0.0 / 0.0),
   a pair of unequal dtype, a pair that is one tensor object, an EMA key without a raw key, an empty pair, optimizer state
   and Python values loads to exactly the saved dict, bit for bit, with torch.load's default, weights_only=True and mmap=True;
   the tensors changed right after the save do not reach the file
T2 size: the file is smaller than torch.save's file of the same dict (independent copies) by the bytes of the equal
   pairs, to 1%; the equal pairs share a storage in the loaded dict and no other pair of keys does
T3 use: load_state_dict of the loaded dict into a module with both sub-modules gives parameters that share no memory
   (a change of one leaves the other); so does a load with assign=True followed by a clone, the held-out tool's pattern
   on a device
T4 moving EMA: with no equal pair the file has the size and the content of the file written by the writer before this
   change (its source is the second argument: `git show ff06bb9:baselines/PixelGen/src/plugins/async_ckpt_io.py`)
T5 ids: tensors that a container makes anew at each visit (so Python may reuse an id) are each written with their own
   values
usage: python test_async_ckpt_twins.py <scratch dir> [<old async_ckpt_io.py>]; exit code = number of failed tests"""
import collections, hashlib, importlib.util, os, sys, traceback
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "baselines", "PixelGen"))
from src.plugins.async_ckpt_io import AsyncShmCheckpointIO

INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def bits(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.numel() == 0:
        return True
    return torch.equal(a.contiguous().view(INT[a.element_size()]), b.contiguous().view(INT[b.element_size()]))


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and bits(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a.keys()) == list(b.keys()) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


EQUAL = ["blocks.%d.w" % i for i in range(12)] + ["blocks.%d.b" % i for i in range(12)] + ["nan", "pos", "scalar"]


def make(seed, moving=False):
    """A checkpoint in the shape of a NO_EMA last.ckpt: denoiser.* and ema_denoiser.* (independent tensors, equal bytes
    unless `moving`), other modules, optimizer state."""
    g = torch.Generator().manual_seed(seed)
    raw = collections.OrderedDict()
    for i in range(12):
        raw[f"blocks.{i}.w"] = torch.randn(256, 256 + i, generator=g)
        raw[f"blocks.{i}.b"] = torch.randn(256, generator=g).bfloat16()
    raw["nan"] = torch.randn(33, generator=g)
    raw["nan"][5] = float("nan")
    raw["pos"] = torch.randn(1, 64, 96, generator=g)
    raw["signbit"] = torch.zeros(17)
    raw["dtype"] = torch.randn(40, generator=g)
    raw["shared"] = torch.randn(50, generator=g)
    raw["empty"] = torch.empty(0, 3)
    raw["scalar"] = torch.tensor(3.5)
    sd = collections.OrderedDict(("denoiser." + k, v) for k, v in raw.items())
    sd["denoiser.rawonly"] = torch.randn(9, generator=g)
    for k, v in raw.items():
        e = v.clone()
        if moving and v.numel() and v.is_floating_point():
            e = (v.float() * 0.999 + 0.001).to(v.dtype)
        sd["ema_denoiser." + k] = e
    sd["ema_denoiser.signbit"] = -torch.zeros(17)                   # equal as numbers, one bit apart
    sd["ema_denoiser.dtype"] = raw["dtype"].double()                # another dtype: never a twin
    sd["ema_denoiser.shared"] = sd["denoiser.shared"]               # one object under two keys
    sd["ema_denoiser.emaonly"] = torch.randn(7, generator=g)
    sd["conditioner.w"] = torch.randn(31, 5, generator=g)
    sd["count"] = torch.tensor(7)
    opt = {"state": {i: {"momentum_buffer": torch.randn(256, 256 + i, generator=g)} for i in range(12)},
           "param_groups": [{"lr": 1e-4, "params": list(range(12))}]}
    return {"state_dict": sd, "optimizer_states": [opt], "global_step": seed, "hparams": {"name": "t"}}


def save(io_cls, ck, path):
    io_ = io_cls()
    io_.save_checkpoint(ck, path)
    for t in ck["state_dict"].values():   # training goes on at once
        if t.is_floating_point() and t.numel():
            t.add_(1000.0)
    io_.teardown()


def ptr(t):
    return t.untyped_storage().data_ptr() if t.numel() else None


def t1(d, old):
    p = os.path.join(d, "t1.ckpt")
    save(AsyncShmCheckpointIO, make(1), p)
    ref = make(1)
    for kw in ({"weights_only": False}, {"weights_only": True}, {"weights_only": False, "mmap": True}, {"weights_only": True, "mmap": True}):
        got = torch.load(p, map_location="cpu", **kw)
        assert same(got, ref), f"content differs with {kw}"
    return "4 loads equal the saved dict bit for bit"


def t2(d, old):
    p, q = os.path.join(d, "t1.ckpt"), os.path.join(d, "t2_ref.ckpt")
    ref = make(1)
    torch.save(ref, q)
    twin_bytes = sum(ref["state_dict"]["denoiser." + k].numel() * ref["state_dict"]["denoiser." + k].element_size() for k in EQUAL)
    obj_bytes = ref["state_dict"]["denoiser.shared"].numel() * 4   # torch.save writes one object once too
    a, b = os.path.getsize(p), os.path.getsize(q)
    assert abs((b - a) - twin_bytes) < 0.01 * b, f"sizes {a} {b}, equal pairs {twin_bytes} bytes"
    got = torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
    groups = collections.defaultdict(list)
    for k, v in got.items():
        if v.numel():
            groups[ptr(v)].append(k)
    shared = sorted(tuple(sorted(v)) for v in groups.values() if len(v) > 1)
    want = sorted(tuple(sorted(("denoiser." + k, "ema_denoiser." + k))) for k in EQUAL + ["shared"])
    assert shared == want, f"shared storages {shared}"
    return f"file {a} bytes against {b} (torch.save), equal pairs {twin_bytes} bytes; {len(want)} pairs share a storage, no other keys do"


class Net(torch.nn.Module):
    def __init__(self, sd, prefix):
        super().__init__()
        self.p = torch.nn.ParameterDict({k[len(prefix):].replace(".", "_"): torch.nn.Parameter(torch.zeros_like(v), requires_grad=False)
                                         for k, v in sd.items() if k.startswith(prefix) and v.is_floating_point()})


def t3(d, old):
    got = torch.load(os.path.join(d, "t1.ckpt"), map_location="cpu", weights_only=False)["state_dict"]
    ref = make(1)["state_dict"]
    nets = {}
    for pre in ("denoiser.", "ema_denoiser."):
        nets[pre] = Net(got, pre)
        nets[pre].load_state_dict({"p." + k[len(pre):].replace(".", "_"): v for k, v in got.items() if k.startswith(pre) and v.is_floating_point()}, strict=True)
    a, b = nets["denoiser."].p, nets["ema_denoiser."].p
    for k in EQUAL:
        k = k.replace(".", "_")
        assert ptr(a[k]) != ptr(b[k]), f"{k}: the two modules share memory"
    a["blocks_0_w"].data.add_(1.0)
    assert bits(b["blocks_0_w"].data, ref["ema_denoiser.blocks.0.w"]), "a change of the raw module reached the EMA module"
    # the held-out tool's pattern: assign=True (the module holds the loaded tensors), then a copy to the device
    n2 = {pre: Net(got, pre) for pre in ("denoiser.", "ema_denoiser.")}
    for pre, n in n2.items():
        n.load_state_dict({"p." + k[len(pre):].replace(".", "_"): v for k, v in got.items() if k.startswith(pre) and v.is_floating_point()}, strict=True, assign=True)
    c = {pre: {k: v.detach().clone() for k, v in n.p.items()} for pre, n in n2.items()}
    for k in EQUAL:
        k2 = k.replace(".", "_")
        assert bits(c["denoiser."][k2], ref["denoiser." + k]) and bits(c["ema_denoiser."][k2], ref["ema_denoiser." + k]), k
        assert ptr(c["denoiser."][k2]) != ptr(c["ema_denoiser."][k2])
    return "load_state_dict gives independent parameters; assign=True then a copy gives the saved values"


def t4(d, old):
    if not old:
        return "SKIPPED (no old source given)"
    spec = importlib.util.spec_from_file_location("old_async_ckpt_io", old)
    sys.path.insert(0, os.path.dirname(os.path.abspath(old)))   # the spawned writer imports the module by its name
    mod = importlib.util.module_from_spec(spec)
    sys.modules["old_async_ckpt_io"] = mod
    spec.loader.exec_module(mod)
    out = []
    for name, moving in (("moving", True), ("twins", False)):
        p, q = os.path.join(d, f"t4_{name}_new.ckpt"), os.path.join(d, f"t4_{name}_old.ckpt")
        save(AsyncShmCheckpointIO, make(4, moving), p)
        save(mod.AsyncShmCheckpointIO, make(4, moving), q)
        a, b = torch.load(p, weights_only=False), torch.load(q, weights_only=False)
        assert same(a, b) and same(a, make(4, moving)), f"{name}: content differs from the old writer's file"
        sa, sb = os.path.getsize(p), os.path.getsize(q)
        md5 = [hashlib.md5(open(x, "rb").read()).hexdigest()[:12] for x in (p, q)]
        if moving:
            shared_obj = 50 * 4   # the one object under two keys: the old writer copied it twice
            assert abs(sb - sa - shared_obj) <= 512, f"moving EMA: sizes {sa} {sb}"
        else:
            assert sa < 0.75 * sb, f"twins: sizes {sa} {sb}"
        out.append(f"{name}: content equal, {sa} against {sb} bytes (old), md5 {md5[0]} / {md5[1]}")
    return "; ".join(out)


def t5(d, old):
    g = torch.Generator().manual_seed(5)
    ck = {"state_dict": collections.OrderedDict((f"k{i}", torch.randn(64, 64, generator=g)) for i in range(64)), "global_step": 5}
    import lightning.fabric.utilities.apply_func as af
    orig = af.apply_to_collection
    # every visited tensor is replaced by a new object that dies when the visit ends: its id is free for the next one
    def hostile(data, dtype, function, *a, **k):
        return orig(data, dtype, lambda t: function(t.clone()), *a, **k)
    import src.plugins.async_ckpt_io as m
    m.apply_to_collection = hostile
    try:
        p = os.path.join(d, "t5.ckpt")
        io_ = AsyncShmCheckpointIO()
        io_.save_checkpoint(ck, p)
        io_.teardown()
    finally:
        m.apply_to_collection = orig
    got = torch.load(p, weights_only=False)
    assert same(got, ck), "tensors made anew at each visit were mixed up"
    n = len({ptr(v) for v in got["state_dict"].values()})
    assert n == 64, f"{n} storages for 64 tensors"
    return "64 short-lived tensors, each written with its own values"


def main():
    d = sys.argv[1]
    old = sys.argv[2] if len(sys.argv) > 2 else None
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        if f.endswith(".ckpt") or ".tmp" in f:
            os.remove(os.path.join(d, f))
    bad = 0
    for t in (t1, t2, t3, t4, t5):
        try:
            print(f"{t.__name__} PASS: {t(d, old)}", flush=True)
        except Exception:
            bad += 1
            print(f"{t.__name__} FAIL:\n{traceback.format_exc()}", flush=True)
    left = [f for f in os.listdir(d) if ".tmp" in f]
    if left:
        bad += 1
        print(f"FAIL: temporary files left {left}")
    print("ok" if not bad else f"FAIL ({bad})")
    sys.exit(bad)


if __name__ == "__main__":
    main()
