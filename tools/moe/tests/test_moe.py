#!/usr/bin/env python3
"""CPU tests of the MoE feed-forward (src/models/layers/moe.py, branch moe), fp32:
 1. one expert, one active = the dense SwiGLU with the same weights (training and evaluation paths);
 2. every route's slot table: the layer's output equals a token-by-token loop over the slots, and the table has the
    route's properties (tc: G experts per token, weights sum to G; ecb: exactly G M / n tokens per expert, its highest
    scores; eci: the top G T / n tokens of every image; noise, pos, hash: the fixed rule);
 3. evaluation does not depend on the batch: an image alone gives the same output as inside a batch (all routes);
 4. the router gets a finite, nonzero gradient (tc, ecb, eci);
 5. the balance callback: tc's bias moves by rate * sign(mean - load) and lowers MaxVio of a skewed router; ecb's
    evaluation thresholds are set from the boundary scores and select about G M / n tokens per expert;
 6. the routing state is in the state dict (moe_bias, moe_thr, moe_thr_set), the statistics are not;
 7. Muon: the experts' row units, the coefficients sqrt(h / d) for e12 and sqrt(d / h) / G for e3t (transposed rows), the
    router with AdamW; one step equals -lr * coef * msign(gradient) per unit;
 8. JiT_T2I with one expert per image block = the dense JiT_T2I with the same weights; a dense JiT_T2I has no MoE keys;
 9. JiT_T2I forward + backward with every route, router_in = plain, a shared expert, dense first blocks, every 2nd block.
usage (baselines/PixelGen on sys.path by this file): python test_moe.py"""
import copy, math, os, sys
import torch
import torch.nn.functional as F

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen"))
sys.path.insert(0, SRC)
from src.models.layers.moe import MoEFeedForward   # noqa: E402
from src.models.layers.swiglu import SwiGLU   # noqa: E402

torch.manual_seed(0)
OK = True


def check(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(f"{'ok  ' if cond else 'FAIL'} {name} {detail}", flush=True)


def loop(layer, xf, idx, gate):
    """The slot table applied token by token."""
    n, h, d = layer.n, layer.h, layer.dim
    W12, W3t = layer.e12.view(n, 2 * h, d), layer.e3t.view(n, h, d)
    y = torch.zeros_like(xf)
    for e in range(n):
        for c in range(idx.shape[1]):
            w = 1.0 if gate is None else float(gate[e, c])
            if w == 0.0:
                continue
            i = int(idx[e, c])
            a = W12[e] @ xf[i]
            y[i] += w * (W3t[e].T @ (F.silu(a[:h]) * a[h:]))
    return y


d, H, B, T = 32, 128, 8, 16
x = torch.randn(B, T, d)
t = torch.rand(B)

# ---- 1. dense equivalence ----
dense = SwiGLU(d, H)
for p in dense.parameters():
    torch.nn.init.normal_(p, std=0.1)
one = MoEFeedForward(d, H, experts=1, active=1, route="tc", tokens=T)
torch.nn.init.normal_(one.router.weight, std=0.5)
with torch.no_grad():
    one.e12.copy_(dense.w12.weight); one.e3t.copy_(dense.w3.weight.t())
    ref = dense(x)
    one.train(); y_tr = one(x)
    one.eval(); y_ev = one(x)
check("1 dense = 1 expert (training path)", torch.equal(ref, y_tr), f"max |diff| {float((ref - y_tr).abs().max()):.1e}")
check("1 dense = 1 expert (evaluation path)", (ref - y_ev).abs().max() < 1e-6, f"max |diff| {float((ref - y_ev).abs().max()):.1e}")

# ---- 2. slot tables ----
def make(route, n=8, G=2, **kw):
    m = MoEFeedForward(d, H, experts=n, active=G, route=route, tokens=T, **kw)
    if m.router is not None:
        torch.nn.init.normal_(m.router.weight, std=0.5)
    return m.train()

cases = [("tc", make("tc", capacity=4.0)), ("tc capacity 1.0", make("tc", capacity=1.0)), ("ecb", make("ecb")), ("eci", make("eci")),
         ("noise", make("noise", n=4, G=1, t_edges=[0.25, 0.5, 0.75], capacity=4.0)), ("pos", make("pos", G=1)), ("hash", make("hash", G=1))]
for name, m in cases:
    with torch.no_grad():
        m.moe_stat.zero_()
        idx, gate = m.slots(x, t)
        dropped = float(m.moe_stat[2 * m.n])   # of this one call
        y = m(x, t)
        yl = loop(m, x.reshape(-1, d), idx, gate).view(B, T, d)
    check(f"2 {name}: output = loop over the slots", (y - yl).abs().max() < 1e-5, f"max |diff| {float((y - yl).abs().max()):.1e}")
    M, n, G = B * T, m.n, m.G
    if m.route in ("tc", "ecb", "eci"):
        with torch.no_grad():
            sig = torch.sigmoid(m._scores(x, None, None))
        wsum = torch.zeros(M).index_add_(0, idx.reshape(-1), gate.reshape(-1))
        cnt = torch.zeros(M).index_add_(0, idx.reshape(-1), (gate > 0).float().reshape(-1))
    if name == "tc":
        check("2 tc: G experts per token, weights sum to G", bool((cnt == G).all()) and (wsum - G).abs().max() < 1e-5,
              f"dropped {dropped:.0f}")
        top = sig.topk(G, dim=1).indices
        got = torch.zeros(M, n).index_put_((idx.reshape(-1), torch.arange(n).repeat_interleave(idx.shape[1])),
                                           (gate > 0).float().reshape(-1), accumulate=True) > 0   # empty slots point at token 0
        check("2 tc: the top-G experts by sigmoid(score) + bias", bool(got.gather(1, top).all()) and int(got.sum()) == G * M)
    if name == "tc capacity 1.0":
        per = (gate > 0).sum(1)
        check("2 tc capacity 1.0: at most G M / n tokens per expert, the surplus counted",
              int(per.max()) <= G * M // n and dropped == G * M - int(per.sum()) and dropped > 0,
              f"dropped {dropped:.0f} of {G * M}")
    if name == "ecb":
        per = (gate > 0).sum(1)
        got = torch.zeros(M, n, dtype=torch.bool)
        got[idx.reshape(-1), torch.arange(n).repeat_interleave(idx.shape[1])] = True
        lo = torch.where(got, sig, torch.ones_like(sig)).min(0).values       # the lowest selected score per expert
        hi = torch.where(got, torch.zeros_like(sig), sig).max(0).values      # the highest unselected
        check("2 ecb: exactly G M / n tokens per expert", bool((per == G * M // n).all()))
        check("2 ecb: its highest scores (no unselected token scores above a selected one)", bool((hi <= lo).all()),
              f"max (unselected - selected) {float((hi - lo).max()):.1e}")
        check("2 ecb: weights of a token sum to G (tokens with an expert)", bool(((wsum - G).abs() < 1e-4)[cnt > 0].all()),
              f"tokens without an expert {int((cnt == 0).sum())} of {M}, most experts on one token {int(cnt.max())}")
    if name == "eci":
        c = G * T // n
        s = m._scores(x, None, None).view(B, T, n)
        ok = True
        for b in range(B):
            for e in range(n):
                mine = set((idx[e, b * c:(b + 1) * c] - b * T).tolist())
                ok &= mine == set(s[b, :, e].topk(c).indices.tolist())
        check("2 eci: the top G T / n tokens of every image for every expert", ok)
    if name == "noise":
        e_img = (t[:, None] > torch.tensor([0.25, 0.5, 0.75])[None, :]).sum(1)   # torch.bucketize(t, edges), right=False
        got = torch.full((M,), -1)
        for e in range(n):
            got[idx[e][gate[e] > 0]] = e
        check("2 noise: every token goes to its image's noise bin", bool((got.view(B, T) == e_img[:, None]).all()))
    if name in ("pos", "hash"):
        cnt = torch.zeros(M).index_add_(0, idx.reshape(-1), torch.ones(idx.numel()))
        same = all(torch.equal(idx[:, :T // n], idx[:, b * (T // n):(b + 1) * (T // n)] - b * T) for b in range(B))
        check(f"2 {name}: every token once, the same partition in every image", bool((cnt == 1).all()) and same)

# ---- 3. evaluation does not depend on the batch ----
for name, m in cases:
    m = copy.deepcopy(m).eval()
    if m.route == "ecb":
        with torch.no_grad():
            m.moe_thr.copy_(torch.sigmoid(m._scores(x, None, None)).quantile(0.75, dim=0))
    with torch.no_grad():
        full = m(x, t)
        alone = torch.cat([m(x[i:i + 1], t[i:i + 1]) for i in range(B)])
    check(f"3 {name}: an image alone = inside the batch (evaluation)", (full - alone).abs().max() < 1e-5,
          f"max |diff| {float((full - alone).abs().max()):.1e}")

# ---- 4. router gradient ----
for name in ("tc", "ecb", "eci"):
    m = make(name)
    m(x).square().sum().backward()
    g = m.router.weight.grad
    check(f"4 {name}: the router's gradient is finite and nonzero", bool(torch.isfinite(g).all()) and float(g.abs().max()) > 0
          and bool(torch.isfinite(m.e12.grad).all()) and bool(torch.isfinite(m.e3t.grad).all()), f"max |grad| {float(g.abs().max()):.1e}")

# ---- 5. balance callback ----
from src.callbacks.moe_balance import MoEBalance   # noqa: E402
m = make("tc", capacity=4.0, bias_rate=0.01)
with torch.no_grad():
    m.router.weight[0] += 3.0 * x.mean((0, 1)) / x.mean((0, 1)).norm()     # expert 0 scores high on every token
    m.router.weight[1] -= 3.0 * x.mean((0, 1)) / x.mean((0, 1)).norm()
cb = MoEBalance()
cb.layers, cb.names = [m], ["l0"]
xs = torch.randn(64, T, d) + x.mean((0, 1))
with torch.no_grad():
    m(xs)
    load = m.moe_stat[:m.n].clone()
    b0 = m.moe_bias.clone()
    st = cb.update()
check("5 tc: bias += rate * sign(mean load - load); statistics zeroed",
      torch.allclose(m.moe_bias - b0, 0.01 * torch.sign(load.mean() - load)) and float(m.moe_stat.abs().sum()) == 0
      and torch.equal(st[0, :m.n], load))
v0 = MoEBalance.summary(st[0], m.n, m.G)["maxvio"]
with torch.no_grad():
    for _ in range(400):
        m(xs); st = cb.update()
v1 = MoEBalance.summary(st[0], m.n, m.G)["maxvio"]
check("5 tc: the bias lowers MaxVio of a skewed router", v1 < 0.5 * v0, f"MaxVio {v0:.2f} -> {v1:.2f}")
m = make("ecb")
cb.layers = [m]
with torch.no_grad():
    m(xs); cb.update()
    m.eval()
    idx, gate = m.slots_eval(xs)
    per = (gate > 0).sum(1).float()
capn = m.G * xs.shape[0] * T // m.n
check("5 ecb: evaluation thresholds set from the boundary scores", bool(m.moe_thr_set) and bool(((per - capn).abs() <= 0.05 * capn + 4).all()),
      f"tokens per expert in evaluation {int(per.min())} to {int(per.max())} (training: {capn})")

# ---- 6. state dict ----
sd = m.state_dict()
check("6 the routing state is saved, the statistics are not",
      {"moe_bias", "moe_thr", "moe_thr_set"} <= set(sd) and "moe_stat" not in sd, str(sorted(sd)))
m2 = MoEFeedForward(d, H, experts=8, active=2, route="ecb", tokens=T)
m2.load_state_dict(sd)
check("6 a loaded layer routes the same", torch.equal(m2.moe_thr, m.moe_thr) and bool(m2.moe_thr_set))

# ---- 7. Muon units ----
from src.models.transformer.JiT_T2I import JiT_T2I   # noqa: E402
from src.optim.muon import MuonAdamW, tag_muon_params, msign   # noqa: E402
SMALL = dict(input_size=64, in_channels=3, num_groups=4, hidden_size=128, num_blocks=3, num_text_blocks=1, patch_size=16,
             txt_embed_dim=64, txt_max_length=16, bottleneck_dim=32)
net = JiT_T2I(**SMALL, moe_experts=4, moe_active=2, moe_route="tc")
tag_muon_params(net)
P = dict(net.named_parameters())
e12, e3t, rt = P["blocks.1.mlp.e12"], P["blocks.1.mlp.e3t"], P["blocks.1.mlp.router.weight"]
hd, dd, G = net.blocks[1].mlp.h, 128, 2
check("7 units: e12 2n row units, e3t n transposed units with factor 1 / G, the router with AdamW",
      e12._muon_parts == 8 and e3t._muon_parts == 4 and e3t._muon_T and e3t._muon_mult == 0.5 and not e12._muon_T
      and not hasattr(rt, "_muon_parts") and P["blocks.1.attn.qkv_x.weight"]._muon_parts == 3)
lr = 1e-2
opt = MuonAdamW([{"params": list(net.parameters())}], lr=1e-3, muon_lr=lr, split=True, ns_dtype="float32", update_scale="mup")
g = torch.Generator().manual_seed(5)
for p in net.parameters():
    p.grad = torch.randn(p.shape, generator=g) * 1e-2
w12, w3, r0 = e12.detach().clone(), e3t.detach().clone(), rt.detach().clone()
opt.step()
k = 2   # expert 2: its gate rows of e12, and its e3t rows
u12 = msign(e12.grad[k * 2 * hd:k * 2 * hd + hd][None], dtype=torch.float32)[0]
u3 = msign(e3t.grad[k * hd:(k + 1) * hd][None], dtype=torch.float32)[0]
d12 = (e12.detach() - w12)[k * 2 * hd:k * 2 * hd + hd]
d3 = (e3t.detach() - w3)[k * hd:(k + 1) * hd]
e1 = float((d12 + lr * math.sqrt(hd / dd) * u12).abs().max()); e2 = float((d3 + lr * math.sqrt(dd / hd) / G * u3).abs().max())
check("7 one step: e12 unit -lr sqrt(h / d) msign(g); e3t unit -lr sqrt(d / h) / G msign(g)", e1 < 1e-6 and e2 < 1e-6,
      f"max |diff| {e1:.1e}, {e2:.1e}; |update| {float(d12.abs().max()):.1e}, {float(d3.abs().max()):.1e}")
check("7 the router moved by AdamW", 0 < float((rt.detach() - r0).abs().max()) <= 1.01e-3)
net = JiT_T2I(**SMALL, moe_experts=4, moe_active=2, moe_lr_mult=0.5)
tag_muon_params(net)
P = dict(net.named_parameters())
check("7 moe_lr_mult scales the experts' coefficients only", P["blocks.0.mlp.e12"]._muon_mult == 0.5 and P["blocks.0.mlp.e3t"]._muon_mult == 0.25
      and not hasattr(P["blocks.0.attn.proj.weight"], "_muon_mult"))
net = JiT_T2I(**SMALL, moe_experts=4, moe_active=1, moe_route="noise")
tq = torch.sigmoid(-0.8 + 0.8 * torch.randn(200000))
share = torch.bincount(torch.bucketize(tq, net.blocks[0].mlp.moe_edges), minlength=4).float() / 200000
check("7 noise route: the default edges give each expert a quarter of the training timesteps", bool((share - 0.25).abs().max() < 0.005),
      f"edges {[round(v, 4) for v in net.blocks[0].mlp.moe_edges.tolist()]}, shares {[round(v, 3) for v in share.tolist()]}")

# ---- 8. model level ----
torch.manual_seed(1)
dn = JiT_T2I(**SMALL)
for p in dn.parameters():
    if p.requires_grad:
        p.data.add_(torch.randn_like(p) * 0.05)
mo = JiT_T2I(**SMALL, moe_experts=1, moe_active=1, moe_route="tc")
sd = dn.state_dict()
check("8 a dense JiT_T2I has no MoE keys", not any("moe" in k or "e12" in k or "router" in k for k in sd) and dn.moe_blocks == [])
for i in range(SMALL["num_blocks"]):
    sd[f"blocks.{i}.mlp.e12"] = sd.pop(f"blocks.{i}.mlp.w12.weight")
    sd[f"blocks.{i}.mlp.e3t"] = sd.pop(f"blocks.{i}.mlp.w3.weight").t().contiguous()
missing = mo.load_state_dict(sd, strict=False)
check("8 load: only the router and routing state are new", all("router" in k or "moe_" in k for k in missing.missing_keys) and not missing.unexpected_keys)
xi, ti, yi = torch.randn(4, 3, 64, 64), torch.rand(4), torch.randn(4, 16, 64)
with torch.no_grad():
    a = dn.train()(xi, ti, yi); b = mo.train()(xi, ti, yi); c = mo.eval()(xi, ti, yi)
check("8 JiT_T2I, 1 expert per block = dense (training, evaluation)", (a - b).abs().max() < 1e-5 and (a - c).abs().max() < 1e-5,
      f"max |diff| {float((a - b).abs().max()):.1e}, {float((a - c).abs().max()):.1e}")

# ---- 9. every option runs ----
EDGES = [0.25, 0.5, 0.75]
for kw in (dict(moe_route="tc"), dict(moe_route="ecb"), dict(moe_route="eci"), dict(moe_route="noise", moe_active=1, moe_t_edges=EDGES),
           dict(moe_route="pos", moe_active=1), dict(moe_route="hash", moe_active=1), dict(moe_route="tc", moe_router_in="plain"), dict(moe_route="tc", moe_router_in="modt"),
           dict(moe_route="tc", moe_router_in="modt", moe_shared_hidden=128),
           dict(moe_route="tc", moe_shared_hidden=128), dict(moe_route="ecb", moe_first_dense=1), dict(moe_route="tc", moe_every=2)):
    net = JiT_T2I(**SMALL, moe_experts=4, **{"moe_active": 2, **kw})
    for p in net.final_layer.parameters():   # the final layer starts at zero: give the blocks a gradient
        torch.nn.init.normal_(p, std=0.02)
    for blk in net.blocks:
        torch.nn.init.normal_(blk.adaLN_modulation[-1].weight, std=0.02)
    out = net.train()(xi, ti, yi)
    out.square().mean().backward()
    gr = [p.grad for n_, p in net.named_parameters() if ".mlp.e" in n_]
    ev = net.eval()(xi, ti, yi)
    check(f"9 {kw}: blocks {net.moe_blocks}", bool(torch.isfinite(out).all()) and bool(torch.isfinite(ev).all())
          and all(g_ is not None and bool(torch.isfinite(g_).all()) and float(g_.abs().max()) > 0 for g_ in gr))


# ---- 10. router_in = modt: the router scores the modulated input plus the timestep embedding (owner 2026-11-19) ----
lay_m = make("tc", n=8, G=2, router_in="mod"); lay_t = make("tc", n=8, G=2, router_in="modt"); lay_p = make("tc", n=8, G=2, router_in="plain")
lay_t.router.weight.data.copy_(lay_m.router.weight.data); lay_p.router.weight.data.copy_(lay_m.router.weight.data)
x10, c10, xn10 = torch.randn(2, 6, lay_m.dim), torch.randn(2, 1, lay_m.dim), torch.randn(2, 6, lay_m.dim)
s_m, s_t, s_p = lay_m._scores(x10, None, c10), lay_t._scores(x10, None, c10), lay_p._scores(x10, xn10, c10)
want = torch.nn.functional.linear((x10 + c10).reshape(-1, lay_m.dim), lay_m.router.weight)
check("10 modt scores = W (x + cond); mod unchanged (W x); plain unchanged (W (xn + cond))",
      (s_t - want).abs().max() < 1e-5 and (s_m - torch.nn.functional.linear(x10.reshape(-1, lay_m.dim), lay_m.router.weight)).abs().max() < 1e-5
      and (s_p - torch.nn.functional.linear((xn10 + c10).reshape(-1, lay_m.dim), lay_m.router.weight)).abs().max() < 1e-5,
      f"max |diff| {float((s_t - want).abs().max()):.1e}")
c10r = c10.clone().requires_grad_(True)
lay_t._scores(x10, None, c10r).sum().backward()
check("10 modt: the timestep embedding gets a gradient through the router", c10r.grad is not None and float(c10r.grad.abs().sum()) > 0)

print("MOE_TEST " + ("PASS" if OK else "FAIL"), flush=True)
sys.exit(0 if OK else 1)
