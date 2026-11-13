"""CPU tests of ARTM on the final ladder's code (branch artm-ladder; docs/artm_ladder.md). A vis node:
    bash tools/gmflow_ladder/vis.sh tools/artm_ladder/test_artm.py
A1  init: every backbone tensor of ARTMJiT equals the point model's at the same seed; FinalLayer keeps no parameter; the new names are art_head.*,
    art_bos, art_seg; the CPU RNG ends in the point model's state.
A2  the mask: a target output at position i does not change when target inputs at positions >= i change (it reads patches < i only), and does change
    when the context or an earlier target patch changes; with a caption mask, padded caption tokens change nothing.
A3  the KV cache: teacher-forced artm_sample gives, at every transition and position, the tokens z of the full masked forward on the same inputs
    (to float rounding), with and without a caption mask, and its final image is the teacher's last state.
A4  ARTMTrainer: X_k and X_{k+1} follow the independent linear process (the draws by hand, k in {0, 1, 2}, X_T = x); the loss equals the hand
    computation; gradients reach the head, the blocks, art_bos and art_seg.
A5  artm_sample with the head: an image of the input's shape, finite; two seeds differ; the same seed repeats.
A6  the CPU training harness (tools/gmflow_ladder/harness.py, artm): finite losses over 8 steps, EMA off with the EMA slot equal to the raw weights,
    every parameter in exactly one group, an exact resume.
A7  Muon marks: the same as the point model's; none on art_*.
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "baselines", "PixelGen"))
from src.models.transformer.JiT_T2I import JiT_T2I   # noqa: E402
from src.models.transformer.ARTM_JiT import ARTMJiT   # noqa: E402
from src.diffusion.flow_matching.training_artm import ARTMTrainer   # noqa: E402
from src.optim.muon import tag_muon_params   # noqa: E402

SMALL = dict(input_size=64, in_channels=3, num_groups=4, hidden_size=64, num_blocks=3, num_text_blocks=1, patch_size=16, txt_embed_dim=32,
             txt_max_length=32, bottleneck_dim=16)
ART = dict(art_T=3, art_head_width=48, art_head_depth=2, art_head_steps=4)
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name} {detail}", flush=True)


def build(seed, cls=JiT_T2I, **kw):
    torch.manual_seed(seed)
    return cls(**SMALL, **kw)


def perturb(net, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            if p.requires_grad:
                p.add_(0.05 * torch.randn(p.shape, generator=g))
    return net


# ---- A1 ----
point = build(0)
st_p = torch.get_rng_state()
art = build(0, ARTMJiT, **ART)
st_a = torch.get_rng_state()
sp, sa = point.state_dict(), art.state_dict()
back = [k for k in sp if not k.startswith("final_layer.linear.")]
check("A1 every backbone tensor equals the point model's init", all(torch.equal(sp[k], sa[k]) for k in back), f"({len(back)} tensors)")
new = sorted({k.split(".")[0] for k in sa if k not in sp})
check("A1 FinalLayer keeps no parameter; the new names are art_head, art_bos, art_seg",
      not any(k.startswith("final_layer.") for k in sa) and new == ["art_bos", "art_head", "art_seg"], str(new))
check("A1 the CPU RNG ends in the point model's state", torch.equal(st_p, st_a))

# ---- A2 ----
net = perturb(build(0, ARTMJiT, **ART), 3).eval()
B, C, H, W = 2, 3, 64, 64
N, P = 16, C * 16 * 16
ctx, tgt = torch.randn(B, N, P), torch.randn(B, N, P)
t = torch.tensor([0.0, 2 / 3])
y = torch.randn(B, 32, 32)
with torch.no_grad():
    z0 = net(ctx, t, y, art_tgt=tgt)
    i = 6
    tgt2 = tgt.clone(); tgt2[:, i:] = torch.randn(B, N - i, P)   # inputs at positions >= i + 1 read patches >= i
    z1 = net(ctx, t, y, art_tgt=tgt2)
    same_upto = (z0[:, :i + 1] - z1[:, :i + 1]).abs().max().item()
    later = (z0[:, i + 1:] - z1[:, i + 1:]).abs().max().item()
    tgt3 = tgt.clone(); tgt3[:, i - 1] += 1.0                     # patch i - 1 is the input at position i
    z3 = net(ctx, t, y, art_tgt=tgt3)
    ctx4 = ctx.clone(); ctx4[:, -1] += 1.0
    z4 = net(ctx4, t, y, art_tgt=tgt)
check("A2 position i reads target patches < i only", same_upto == 0.0 and later > 1e-4, f"(<= i: {same_upto:.1e}, > i: {later:.2e})")
check("A2 an earlier target patch and the context reach position i",
      (z3[:, i] - z0[:, i]).abs().max().item() > 1e-4 and (z3[:, :i] - z0[:, :i]).abs().max().item() == 0.0 and (z4[:, 0] - z0[:, 0]).abs().max().item() > 1e-4)
ym = torch.zeros(B, 10, dtype=torch.bool); ym[0, :4] = True; ym[1, :7] = True
yy = torch.randn(B, 10, 32)
with torch.no_grad():
    za = net(ctx, t, yy, y_mask=ym, art_tgt=tgt)
    yb = yy.clone(); yb[0, 4:] = torch.randn(6, 32); yb[1, 7:] = torch.randn(3, 32)
    zb = net(ctx, t, yb, y_mask=ym, art_tgt=tgt)
check("A2 padded caption tokens change nothing", torch.equal(za, zb))

# ---- A3 ----
for masked in (False, True):
    with torch.no_grad():
        x0 = torch.randn(B, C, H, W)
        teach = [torch.randn(B, N, P) for _ in range(3)]
        kw = {"y_mask": ym} if masked else {}
        yin = yy if masked else y
        img, zs = net.artm_sample(x0, yin, teacher=teach, **kw)
        worst = 0.0
        xk = x0
        for k in range(3):
            ctxk = torch.nn.functional.unfold(xk, kernel_size=16, stride=16).transpose(1, 2)
            zf = net(ctxk, torch.full((B,), k / 3), yin, art_tgt=teach[k], **kw)
            worst = max(worst, (zf - zs[k]).abs().max().item())
            xk = net.unpatchify(teach[k], H, W)
    check(f"A3 the KV cache equals the full masked forward ({'caption mask' if masked else 'no caption mask'})",
          worst < 1e-4 and torch.allclose(img, net.unpatchify(teach[2], H, W)), f"(max |z diff| over 3 x {N} steps {worst:.1e})")

# ---- A4 ----
tr = ARTMTrainer(head_P_mean=-0.8, head_P_std=1.6, t_eps=0.05, null_condition_p=0.0)
seen = {}


class Spy(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner, self.patch_size, self.art_T = inner, inner.patch_size, inner.art_T

    def forward(self, c, tt, yy_, **kw):
        seen["c"], seen["t"], seen["kw"] = c.clone(), tt.clone(), kw
        return self.inner(c, tt, yy_, **kw)


x = torch.rand(B, C, H, W) * 2 - 1
torch.manual_seed(21)
out = tr._impl_trainstep(Spy(net), None, None, x, y, {})
torch.manual_seed(21)
k = torch.randint(0, 3, (B,))
ea, eb = torch.randn_like(x), torch.randn_like(x)
sa_, sb_ = (k.float() / 3).view(-1, 1, 1, 1), ((k + 1).float() / 3).view(-1, 1, 1, 1)
xa, xb = (1 - sa_) * ea + sa_ * x, (1 - sb_) * eb + sb_ * x
cref = torch.nn.functional.unfold(xa, kernel_size=16, stride=16).transpose(1, 2)
tref = torch.nn.functional.unfold(xb, kernel_size=16, stride=16).transpose(1, 2)
tau = (torch.randn(B, N) * 1.6 - 0.8).sigmoid()
eta = torch.randn_like(tref)
q = tau[..., None] * tref + (1 - tau[..., None]) * eta
with torch.no_grad():
    hat = net(cref, k.float() / 3, y, art_tgt=tref, art_q=q, art_tau=tau)
lref = (((hat - tref) / (1 - tau[..., None]).clamp_min(0.05)) ** 2).mean()
check("A4 the independent linear process and the loss by hand",
      torch.equal(seen["c"], cref) and torch.equal(seen["kw"]["art_tgt"], tref) and torch.allclose(out["loss"], lref, rtol=1e-5),
      f"(k {k.tolist()}, loss {float(out['loss']):.5f} against {float(lref):.5f})")
k3 = torch.full((B,), 2)
xb3 = (1 - ((k3 + 1).float() / 3).view(-1, 1, 1, 1)) * eb + ((k3 + 1).float() / 3).view(-1, 1, 1, 1) * x
check("A4 X_T is the image", torch.equal(xb3, x))
net.zero_grad()
torch.manual_seed(22)
tr._impl_trainstep(net, None, None, x, y, {})["loss"].backward()
g = {n: float(p.grad.abs().sum()) for n, p in net.named_parameters() if p.grad is not None}
check("A4 gradients reach the head, the blocks, art_bos and art_seg",
      sum(v for n, v in g.items() if n.startswith("art_head.")) > 0 and sum(v for n, v in g.items() if n.startswith("blocks.")) > 0
      and g.get("art_bos", 0) > 0 and g.get("art_seg", 0) > 0)

# ---- A5 ----
with torch.no_grad():
    x0 = torch.randn(B, C, H, W)
    torch.manual_seed(5); a1 = net.artm_sample(x0, y)
    torch.manual_seed(6); a2 = net.artm_sample(x0, y)
    torch.manual_seed(5); a3 = net.artm_sample(x0, y)
check("A5 sampling: an image, finite; seeds differ; the same seed repeats",
      a1.shape == (B, C, H, W) and torch.isfinite(a1).all() and not torch.allclose(a1, a2) and torch.equal(a1, a3))

# ---- A6 (the harness) ----
TMP = tempfile.mkdtemp(prefix="artm_test_", dir=os.environ.get("TMPDIR", "/scratch/gpfs/GROUP/USER/tmp/cc_tmp"))


def run(tag, **cfg):
    cfg = dict(dict(out=f"{TMP}/{tag}.pt", root=f"{TMP}/{tag}", k=0, artm=48, steps=8, seed=0, batch=4, ms_every=2, last_every=4, probe=True), **cfg)
    os.makedirs(cfg["root"], exist_ok=True)
    cj = f"{TMP}/{tag}.json"
    json.dump(cfg, open(cj, "w"))
    env = dict(os.environ, TORCHDYNAMO_DISABLE="1", PYTHONPATH=os.path.join(REPO, "baselines", "PixelGen"))
    r = subprocess.run([sys.executable, "-I", os.path.join(REPO, "tools", "gmflow_ladder", "run.py"), "tools/gmflow_ladder/harness.py", cj],
                       env=env, capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-3000:], flush=True)
        raise SystemExit(f"harness {tag} failed")
    return torch.load(cfg["out"], weights_only=False)


h8 = run("artm")
strip = lambda n: n[len("denoiser."):] if n.startswith("denoiser.") else n
names = [strip(n) for grp in h8["groups"] for n in grp["names"]]
check("A6 finite losses over 8 steps", len(h8["losses"]) == 8 and all(math.isfinite(v) for v in h8["losses"]), str([round(v, 4) for v in h8["losses"]]))
check("A6 EMA off, the EMA slot equals the raw weights in every checkpoint", h8["ema_off"] and h8["ema_equal"] and all(h8["ema_equal"].values()), str(h8["saves"]))
check("A6 every parameter in exactly one group", len(names) == len(set(names)) and "art_bos" in names and "art_seg" in names)
a = run("artm_first", steps=4, total_steps=8)
b = run("artm_resumed", steps=8, total_steps=8, root=f"{TMP}/artm_first", resume=f"{TMP}/artm_first/last.ckpt")
check("A6 resume: the weights and the schedule continue", b["probe_start"] == a["probe_end"] and b["lrs"] == h8["lrs"][4:],
      f"(probe {a['probe_end']} against {b['probe_start']})")

# ---- A7 ----
ref, mod = build(0), build(0, ARTMJiT, **ART)
tag_muon_params(ref)
tag_muon_params(mod)
mr = [n for n, p in ref.named_parameters() if hasattr(p, "_muon_parts")]
mm = [n for n, p in mod.named_parameters() if hasattr(p, "_muon_parts")]
check("A7 the same Muon marks as the point model; none on art_*", mr == mm and not any(n.startswith("art_") for n in mm), f"({len(mm)} marked)")
print("ALL PASS" if ok else "SOME FAILED", flush=True)
sys.exit(0 if ok else 1)
