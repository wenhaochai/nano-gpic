"""CPU tests of the DTM flow head on the final ladder's code (branch dtm-ladder; docs/dtm_ladder.md). A vis node:
    bash tools/gmflow_ladder/vis.sh tools/dtm_ladder/test_dtm.py
D1  dtm_width = 0 builds the model of before (no dtm_head, FinalLayer's linear); with dtm_width > 0 and the same seed every backbone tensor equals
    the point model's init bit for bit, FinalLayer keeps no parameter, and the CPU RNG ends in the point model's state.
D2  forward modes: the default output (the head's mean) is an image, deterministic, and equals the mean over dtm_mean_draws fixed head noises at
    head time 0 computed by hand; "draw" is an image, two draws differ, and equals the head's Euler flow by hand under the same seed; dtm_q/dtm_tau
    return the head's x_hat tokens; a point model refuses dtm_mode.
D3  DTMTrainer: t and x_t are JiTTrainer's draws (same RNG order); the loss equals ((x_hat - patch) / max(1 - tau, t_eps))^2 by hand; gradients
    reach the head and the backbone.
D4  DTMDrawSampler: with a stub net, each step is x + (sigma - sigma') (x0 - x) / sigma, the last step returns x0, and it calls the net with
    dtm_mode "draw" at the grid t_i = i / N; CFG other than 1 is refused.
D5  the CPU training harness (tools/gmflow_ladder/harness.py, dtm): finite losses over 8 steps, EMA off with the EMA slot equal to the raw weights,
    the head in the AdamW groups (no Muon mark on dtm_head.*), an exact resume (probe loss and schedule).
D6  Muon marks and parameter groups: the same Muon marks as the point model; dtm_head.* is the only new name; no parameter of FinalLayer.
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
from src.diffusion.flow_matching.scheduling import LinearScheduler   # noqa: E402
from src.diffusion.flow_matching.training_JiT import JiTTrainer, constant   # noqa: E402
from src.diffusion.flow_matching.training_dtm import DTMTrainer   # noqa: E402
from src.diffusion.flow_matching.sampling_dtm import DTMDrawSampler   # noqa: E402
from src.diffusion.base.guidance import simple_guidance_fn   # noqa: E402
from src.optim.muon import tag_muon_params   # noqa: E402

SMALL = dict(input_size=64, in_channels=3, num_groups=4, hidden_size=64, num_blocks=3, num_text_blocks=1, patch_size=16, txt_embed_dim=32,
             txt_max_length=32, bottleneck_dim=16)
DTM = dict(dtm_width=48, dtm_depth=2, dtm_steps=5, dtm_mean_draws=3)
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name} {detail}", flush=True)


def build(seed, **kw):
    torch.manual_seed(seed)
    return JiT_T2I(**SMALL, **kw)


def perturb(net, seed):
    """Give every weight a value (the head's output layer and the adaLN gates start at zero)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            if p.requires_grad:
                p.add_(0.05 * torch.randn(p.shape, generator=g))
    return net


# ---- D1 ----
point = build(0)
st_point = torch.get_rng_state()
dtm = build(0, **DTM)
st_dtm = torch.get_rng_state()
sp, sd = point.state_dict(), dtm.state_dict()
back = [k for k in sp if not k.startswith("final_layer.linear.")]
check("D1 point model: no dtm_head, FinalLayer has its linear", not hasattr(point, "dtm_head") and isinstance(point.final_layer.linear, torch.nn.Linear))
check("D1 DTM model: every backbone tensor equals the point model's init", all(k in sd and torch.equal(sp[k], sd[k]) for k in back), f"({len(back)} tensors)")
check("D1 DTM model: FinalLayer keeps no parameter; the only new names are dtm_head.*",
      not any(k.startswith("final_layer.") for k in sd) and sorted({k.split(".")[0] for k in sd if k not in sp}) == ["dtm_head"])
check("D1 the CPU RNG ends in the point model's state", torch.equal(st_point, st_dtm))

# ---- D2 ----
torch.manual_seed(1)
net = perturb(build(0, **DTM), 7).eval()
B, C, H, W = 2, 3, 64, 64
x, t, y = torch.randn(B, C, H, W), torch.tensor([0.3, 0.7]), torch.randn(B, 32, 32)   # unmasked: captions padded to txt_max_length
with torch.no_grad():
    m1, m2 = net(x, t, y), net(x, t, y, dtm_mode="mean")
    # by hand: the tokens z, then the mean over the fixed head noises at tau = 0
    zs = {}
    hook = net.final_layer.register_forward_hook(lambda mod, inp, out: zs.__setitem__("z", out))
    net(x, t, y)
    hook.remove()
    z = zs["z"]
    N, P = z.shape[1], C * 16 * 16
    g = torch.Generator().manual_seed(20261113)
    acc = sum(net.dtm_head(torch.randn(B * N, P, generator=g), torch.zeros(B * N), z.reshape(B * N, -1)) for _ in range(DTM["dtm_mean_draws"]))
    ref = net.unpatchify((acc / DTM["dtm_mean_draws"]).view(B, N, P), H, W)
check("D2 the default output is the mean: an image, deterministic, equal to the hand computation",
      m1.shape == (B, C, H, W) and torch.equal(m1, m2) and torch.allclose(m1, ref, atol=1e-6), f"(max diff {float((m1 - ref).abs().max()):.2e})")
with torch.no_grad():
    torch.manual_seed(5); d1 = net(x, t, y, dtm_mode="draw")
    torch.manual_seed(6); d2 = net(x, t, y, dtm_mode="draw")
    torch.manual_seed(5)
    hq = torch.randn(B * N, P)
    grid = torch.linspace(0, 1, DTM["dtm_steps"] + 1)
    for j in range(DTM["dtm_steps"]):
        hat = net.dtm_head(hq, grid[j].expand(B * N), z.reshape(B * N, -1))
        hq = hat if j == DTM["dtm_steps"] - 1 else hq + (grid[j + 1] - grid[j]) * (hat - hq) / (1 - grid[j]).clamp_min(0.05)
    dref = net.unpatchify(hq.view(B, N, P), H, W)
check("D2 draw: an image, two seeds differ, equal to the head's Euler flow by hand",
      d1.shape == (B, C, H, W) and not torch.allclose(d1, d2) and torch.allclose(d1, dref, atol=1e-6), f"(max diff {float((d1 - dref).abs().max()):.2e})")
with torch.no_grad():
    q, tau = torch.randn(B, N, P), torch.rand(B, N)
    tok = net(x, t, y, dtm_q=q, dtm_tau=tau)
    tref = net.dtm_head(q.reshape(B * N, P), tau.reshape(B * N), z.reshape(B * N, -1)).view(B, N, P)
check("D2 dtm_q/dtm_tau return the head's x_hat tokens", tok.shape == (B, N, P) and torch.allclose(tok, tref, atol=1e-6))
try:
    point(x, t, y, dtm_mode="draw")
    refused = False
except AssertionError:
    refused = True
check("D2 a point model refuses dtm_mode", refused)

# ---- D3 ----
targs = dict(scheduler=LinearScheduler(), loss_weight_fn=constant, lognorm_t=True, timeshift=1.0, P_mean=-0.8, P_std=0.8, t_eps=0.05, null_condition_p=0.0)
jt, dt = JiTTrainer(**targs), DTMTrainer(**targs, head_P_mean=-0.8, head_P_std=1.6)
seen = {}


class Spy(torch.nn.Module):
    """Records the first call's x_t and t, then runs the wrapped net."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.patch_size = inner.patch_size

    def forward(self, xt, tt, yy, **kw):
        seen.setdefault(len(seen), (xt.clone(), tt.clone(), kw))
        return self.inner(xt, tt, yy, **kw)


xb = torch.randn(B, C, H, W)
point_net = perturb(build(0), 7)
torch.manual_seed(11); jt._impl_trainstep(Spy(point_net), None, None, xb, y, {})
torch.manual_seed(11); out = dt._impl_trainstep(Spy(net), None, None, xb, y, {})
(xt_j, t_j, _), (xt_d, t_d, kw_d) = seen[0], seen[1]
check("D3 t and x_t are JiTTrainer's draws", torch.equal(t_j, t_d) and torch.equal(xt_j, xt_d))
torch.manual_seed(11)
base_t = (torch.randn(B) * 0.8 - 0.8).sigmoid()
noise = torch.randn_like(xb)
patches = torch.nn.functional.unfold(xb, kernel_size=16, stride=16).transpose(1, 2)
tau = (torch.randn(B, N) * 1.6 - 0.8).sigmoid()
eta = torch.randn_like(patches)
q = tau[..., None] * patches + (1 - tau[..., None]) * eta
with torch.no_grad():
    hat = net(xt_d, t_d, y, dtm_q=q, dtm_tau=tau)
lref = (((hat - patches) / (1 - tau[..., None]).clamp_min(0.05)) ** 2).mean()
check("D3 the loss equals the hand computation", torch.allclose(out["loss"], lref, rtol=1e-5) and torch.equal(kw_d["dtm_tau"], tau),
      f"({float(out['loss']):.6f} against {float(lref):.6f})")
net.zero_grad()
seen.clear()
torch.manual_seed(12)
dt._impl_trainstep(net, None, None, xb, y, {})["loss"].backward()
gh = sum(float(p.grad.abs().sum()) for n, p in net.named_parameters() if n.startswith("dtm_head.") and p.grad is not None)
gb = sum(float(p.grad.abs().sum()) for n, p in net.named_parameters() if n.startswith("blocks.") and p.grad is not None)
check("D3 gradients reach the head and the backbone", gh > 0 and gb > 0, f"(|grad| head {gh:.3g}, blocks {gb:.3g})")

# ---- D4 ----
calls = []


class Stub(torch.nn.Module):
    def __init__(self, x0):
        super().__init__()
        self.x0 = x0

    def forward(self, xx, tt, cond, **kw):
        calls.append((float(tt[0]), kw.get("dtm_mode")))
        return self.x0


smp = DTMDrawSampler(scheduler=LinearScheduler(), w_scheduler=LinearScheduler(), guidance_fn=simple_guidance_fn, num_steps=4, guidance=1.0, rank_seed=False)
x0 = torch.randn(B, C, H, W)
noise = torch.randn(B, C, H, W)
xs, vs = smp._impl_sampling(Stub(x0), noise, y, None)
xr = noise.clone()
grid = [0.0, 0.25, 0.5, 0.75, 1.0]
for a, b_ in zip(grid[:-1], grid[1:]):
    xr = x0 if b_ == 1.0 else xr + ((1 - a) - (1 - b_)) * (x0 - xr) / (1 - a)
check("D4 the straight steps by hand; the last step returns the draw; dtm_mode draw at t_i = i / N",
      torch.allclose(xs[-1], xr) and torch.equal(xs[-1], x0) and [c[1] for c in calls] == ["draw"] * 4
      and all(abs(c[0] - g_) < 1e-6 for c, g_ in zip(calls, grid[:-1])))
try:
    DTMDrawSampler(scheduler=LinearScheduler(), w_scheduler=LinearScheduler(), guidance_fn=simple_guidance_fn, num_steps=4, guidance=2.0,
                   rank_seed=False)._impl_sampling(Stub(x0), noise, y, None)
    cfg_refused = False
except AssertionError:
    cfg_refused = True
check("D4 CFG other than 1 is refused", cfg_refused)

# ---- D5 (the harness) ----
TMP = tempfile.mkdtemp(prefix="dtm_test_", dir=os.environ.get("TMPDIR", "/scratch/gpfs/GROUP/USER/tmp/cc_tmp"))


def run(tag, **cfg):
    cfg = dict(dict(out=f"{TMP}/{tag}.pt", root=f"{TMP}/{tag}", k=0, dtm=48, steps=8, seed=0, batch=4, ms_every=2, last_every=4, probe=True), **cfg)
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


g8 = run("dtm")
strip = lambda n: n[len("denoiser."):] if n.startswith("denoiser.") else n
names = [strip(n) for grp in g8["groups"] for n in grp["names"]]
check("D5 finite losses over 8 steps", len(g8["losses"]) == 8 and all(math.isfinite(v) for v in g8["losses"]), str([round(v, 4) for v in g8["losses"]]))
check("D5 EMA off, the EMA slot equals the raw weights in every checkpoint", g8["ema_off"] and g8["ema_equal"] and all(g8["ema_equal"].values()), str(g8["saves"]))
check("D5 every parameter in exactly one group, the head included", len(names) == len(set(names)) and any(n.startswith("dtm_head.") for n in names))
a = run("dtm_first", steps=4, total_steps=8)
b = run("dtm_resumed", steps=8, total_steps=8, root=f"{TMP}/dtm_first", resume=f"{TMP}/dtm_first/last.ckpt")
check("D5 resume: the weights and the schedule continue", b["probe_start"] == a["probe_end"] and b["lrs"] == g8["lrs"][4:],
      f"(probe {a['probe_end']} against {b['probe_start']})")

# ---- D6 ----
ref, mod = build(0), build(0, **DTM)
tag_muon_params(ref)
tag_muon_params(mod)
mr = [n for n, p in ref.named_parameters() if hasattr(p, "_muon_parts")]
mm = [n for n, p in mod.named_parameters() if hasattr(p, "_muon_parts")]
check("D6 the same Muon marks as the point model; none on the head", mr == mm and not any(n.startswith("dtm_head.") for n in mm), f"({len(mm)} marked)")
print("ALL PASS" if ok else "SOME FAILED", flush=True)
sys.exit(0 if ok else 1)
