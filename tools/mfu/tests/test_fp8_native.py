#!/usr/bin/env python3
"""Native tensorwise FP8 (src/models/layers/fp8_native.py), one GPU.
1. tcast: e4m3 values and the tensor's scale bit for bit the plain-torch quantization (reference()'s).
2. fp8_linear forward against the dequantized-GEMM reference (fp32 accumulation): relative error at fp32/bf16 level; and
   against the bf16 GEMM: the size of the FP8 rounding (reported).
3. Its backward: dx = dy W and dw = dy^T x (the bf16 input), bf16 GEMMs, against fp32 references.
4. SwiGLU + w3: forward against the reference on the model's bf16 product, backward dz against autograd of the bf16
   SwiGLU at the same w3, dw3 against dy^T a_q.
5. Phase 2, the fused MLP (FP8 dgrads): the forward bit for bit phase 1's; the SwiGLU backward's FP8 copy bit for bit the
   tensorwise cast of its own bf16 dz; dx, dw12 and dw3 against fp32 autograd of the bf16 MLP (FP8 dgrad error reported).
6. Phase 3 (delayed scaling): tcast_t's copy and transpose bit for bit tcast's; without history every delayed-scaling cast
   bit for bit its phase 1/2 cast (dcast = tcast, swiglu_dcast = swiglu_tcast, swiglu_bwd_ds's FP8 copy = tcast of its dz);
   with history the cast uses 448 / the max of the recorded amaxes (bit for bit the torch reference) and the roll keeps
   the last L_HIST amaxes; a block of four FP8 GEMMs over 3 steps: step 1 (no history) bit for bit phase 2's, steps 2 and 3
   within 1.25 x phase 2's error against fp32 autograd (4 chained FP8 GEMMs err about 0.1 against fp32 in either phase),
   and the same 3 steps under torch.compile give the eager steps' outputs, gradients and state.
7. Phase 4 (FP8 dgrads and wgrads): each cast's transposed copy = its row-major copy transposed, the row-major copies
   = phase 3's without history; a block's forward = phase 3's at every step, its weight gradient within 2 x phase 3's
   error against fp32 (+0.02), compiled with the C++ wrapper = eager.
usage: python test_fp8_native.py"""
import os, sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen")))
from src.models.layers.fp8_native import tcast, fp8_linear, fp8_swiglu_w3, fp8_mlp, reference, swiglu_tcast, swiglu_bwd, FMAX, E4  # noqa

dev, BF = "cuda", torch.bfloat16
torch.manual_seed(0)
ok = True


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


for M, K, N in ((4096, 640, 1920), (2048, 1280, 5120), (1000, 1792, 1792)):
    x = (torch.randn(M, K, device=dev) * torch.rand(M, 1, device=dev) * 4).to(BF)
    x[3] = 0
    w = torch.randn(N, K, device=dev) * 0.02
    b = torch.randn(N, device=dev) * 0.1
    q, s = tcast(x)
    a = x.float().abs().amax().clamp_min(1e-12)
    rq = (x.float() * (FMAX / a)).clamp(-FMAX, FMAX).to(E4)
    same = torch.equal(q.view(torch.uint8), rq.view(torch.uint8)) and torch.equal(s, (a / FMAX).view(1))
    y = fp8_linear(x, w, b)
    yr = reference(x, w, b)
    yb = (x.float() @ w.float().t() + b).to(BF)
    e_ref, e_bf = rel(y, yr), rel(y, yb)
    xl, wl, bl = x.clone().requires_grad_(), w.clone().requires_grad_(), b.clone().requires_grad_()
    dy = torch.randn(M, N, device=dev).to(BF)
    fp8_linear(xl, wl, bl).backward(dy)
    dx_r = dy.float() @ w.float()
    dw_r = dy.float().t() @ x.float()          # the bf16 input (fp8_linear keeps it for the weight gradient)
    e_dx, e_dw, e_db = rel(xl.grad, dx_r), rel(wl.grad, dw_r), rel(bl.grad, dy.float().sum(0))
    good = same and e_ref < 5e-3 and e_dx < 6e-3 and e_dw < 6e-3 and e_db < 6e-3 and torch.isfinite(y).all()
    ok &= bool(good)
    print(f"linear M={M} K={K} N={N}: tcast bit for bit {same}; y vs dequantized reference {e_ref:.1e}, vs bf16 GEMM {e_bf:.1e} "
          f"(FP8 rounding); dx {e_dx:.1e} dw {e_dw:.1e} db {e_db:.1e}" + ("" if good else "  FAIL"))

for M, C in ((2048, 640), (1024, 1280)):
    H = 4 * C
    z = (torch.randn(M, 2 * H, device=dev) * 2).to(BF)
    w3 = torch.randn(C, H, device=dev) * 0.02
    zl, wl = z.clone().requires_grad_(), w3.clone().requires_grad_()
    y = fp8_swiglu_w3(zl, wl)
    z1, z2 = z.chunk(2, -1)
    a = torch.nn.functional.silu(z1) * z2            # the model's bf16 product
    yr = reference(a, w3)
    yb = (a.float() @ w3.float().t()).to(BF)
    aq, sa = swiglu_tcast(z)
    dy = torch.randn(M, C, device=dev).to(BF)
    y.backward(dy)
    zr = z.clone().float().requires_grad_()
    r1, r2 = zr.chunk(2, -1)
    ((torch.nn.functional.silu(r1) * r2) @ w3.float().t()).backward(dy.float())
    dw_r = dy.float().t() @ (aq.float() * sa)
    e_y, e_b, e_dz, e_dw = rel(y, yr), rel(y, yb), rel(zl.grad, zr.grad), rel(wl.grad, dw_r)
    good = e_y < 5e-3 and e_dz < 1e-2 and e_dw < 6e-3 and torch.isfinite(y).all()
    ok &= bool(good)
    print(f"swiglu+w3 M={M} C={C}: y vs reference {e_y:.1e}, vs bf16 {e_b:.1e}; dz vs fp32 autograd {e_dz:.1e}; dw3 {e_dw:.1e}"
          + ("" if good else "  FAIL"))
for M, C in ((2048, 640), (1024, 1280)):
    H = 4 * C
    x = (torch.randn(M, C, device=dev) * 2).to(BF)
    w12 = torch.randn(2 * H, C, device=dev) * (C ** -0.5)
    w3 = torch.randn(C, H, device=dev) * (H ** -0.5)
    dy = torch.randn(M, C, device=dev).to(BF)
    L = [t.clone().requires_grad_() for t in (x, w12, w3)]
    y2 = fp8_mlp(*L)
    y1 = fp8_swiglu_w3(fp8_linear(x, w12), w3)
    same_fwd = torch.equal(y2, y1)
    y2.backward(dy)
    R = [t.clone().float().requires_grad_() for t in (x, w12, w3)]
    zr = R[0] @ R[1].t()
    a1, a2 = zr.chunk(2, -1)
    ((torch.nn.functional.silu(a1) * a2) @ R[2].t()).backward(dy.float())
    z = (x.float() @ w12.float().t()).to(BF)   # close to the FP8 forward's z; only for the kernel check below
    da = torch.randn(M, H, device=dev).to(BF)
    dz, dzq, sdz = swiglu_bwd(z, da)
    q2, s2 = tcast(dz)
    kern = torch.equal(dzq.view(torch.uint8), q2.view(torch.uint8)) and torch.equal(sdz, s2)
    e_dx, e_w12, e_w3 = rel(L[0].grad, R[0].grad), rel(L[1].grad, R[1].grad), rel(L[2].grad, R[2].grad)
    good = same_fwd and kern and e_dx < 8e-2 and e_w12 < 8e-2 and e_w3 < 8e-2 and torch.isfinite(L[0].grad).all()
    ok &= bool(good)
    print(f"mlp phase 2 M={M} C={C}: forward bit for bit phase 1 {same_fwd}; SwiGLU backward FP8 copy bit for bit {kern}; "
          f"vs fp32 autograd: dx {e_dx:.1e} dw12 {e_w12:.1e} dw3 {e_w3:.1e}" + ("" if good else "  FAIL"))
# 6. phase 3
import src.models.layers.fp8_native as F  # noqa: E402


def bits_eq(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def ref_cast(t, mx):   # the cast with a given amax, in torch: scale = 448 / mx (IEEE fp32), clamp, e4m3
    sc = torch.tensor(FMAX, device=dev) / mx
    return (t.float() * sc).clamp(-FMAX, FMAX).to(E4), (mx / torch.tensor(FMAX, device=dev)).view(1)


w = torch.randn(1920, 640, device=dev) * 0.05
q, sq, qt = F.tcast_t(w)
q0, s0 = tcast(w)
q1, s1 = tcast(w.t().contiguous())
good = bits_eq(q, q0) and bits_eq(qt, q1) and torch.equal(sq, s0) and torch.equal(s1, s0)
ok &= bool(good)
print(f"phase 3 tcast_t: copy and transpose bit for bit tcast's: {good}", flush=True)

F.DS.sites, F.DS.st = 8, None
F.ds_state(dev)
F.ds_reset()
xs = [(torch.randn(4096, 640, device=dev) * (1.0 + k)).to(BF) for k in range(4)]
amax = [x.float().abs().amax() for x in xs]
qa, sa = F.dcast(xs[0], F.DS.st, 0, True)                      # no history: phase 1's cast
qr, sr = tcast(xs[0])
g1 = bits_eq(qa, qr) and torch.equal(sa, sr)
qb, sb = F.dcast(xs[1], F.DS.st, 0, True)                      # history [amax0]: scale 448 / amax0 (xs[1] saturates where larger)
rb, rsb = ref_cast(xs[1], amax[0])
g2 = bits_eq(qb, rb) and torch.equal(sb, rsb)
qc, sc_ = F.dcast(xs[2], F.DS.st, 0, True)                     # history [amax1, amax0]: the max is amax1
rc, rsc = ref_cast(xs[2], max(amax[0], amax[1]))
g3 = bits_eq(qc, rc) and torch.equal(sc_, rsc)
st = F.DS.st
hist_ok = torch.equal(st[0, :2], torch.stack([amax[1], amax[0]])) and float(st[0, F.L_HIST]) == float(amax[2])
z = (torch.randn(1024, 2 * 2560, device=dev) * 2).to(BF)
F.ds_reset()
za, zsa = F.swiglu_dcast(z, F.DS.st, 1)
zr, zsr = swiglu_tcast(z)
g4 = bits_eq(za, zr) and torch.equal(zsa, zsr)
F.dcast(xs[0], F.DS.st, 0, True)                               # roll: site 1 gets the product's amax as its history
pm = F.DS.st[1, 0].clone()                            # the kernel's amax of the product (torch's sigmoid can differ by an ulp)
z1, z2 = z.float().chunk(2, -1)
prod = ((z1 * torch.sigmoid(z1)).to(BF).float() * z2).to(BF)
zb, zsb = F.swiglu_dcast(z, F.DS.st, 1)
rzb, rzsb = ref_cast(prod, pm)
mis = float((zb.view(torch.uint8) != rzb.view(torch.uint8)).float().mean())
g5 = mis < 1e-3 and torch.equal(zsb, rzsb)
da = torch.randn(1024, 2560, device=dev).to(BF)
F.ds_reset()
dz, dzq, dzs = F.swiglu_bwd_ds(z, da, F.DS.st, 2)
qd, sd = tcast(dz)
g6 = bits_eq(dzq, qd) and torch.equal(dzs, sd)
F.dcast(xs[0], F.DS.st, 0, True)
dzm = dz.float().abs().amax()
dz2, dzq2, dzs2 = F.swiglu_bwd_ds(z, da, F.DS.st, 2)
rdz, rdzs = ref_cast(dz2, dzm)
g7 = torch.equal(dz2, dz) and bits_eq(dzq2, rdz) and torch.equal(dzs2, rdzs)
good = g1 and g2 and g3 and hist_ok and g4 and g5 and g6 and g7
ok &= bool(good)
print(f"phase 3 casts: dcast without history = tcast {g1}; with history (448 / max of the history) {g2} {g3}; history "
      f"{hist_ok}; swiglu_dcast = swiglu_tcast {g4}, with history {g5} ({mis:.1e} of bytes differ from torch's product); "
      f"swiglu_bwd_ds's copy = tcast(dz) {g6}, with history "
      f"{g7}", flush=True)


class Blk(torch.nn.Module):   # an image block's FP8 GEMMs: phase 2, 3 (sites: qkv_x 0 with the roll, proj 1, MLP 2-5) or 4 (0-7)
    def __init__(self, C, phase):
        super().__init__()
        self.phase = phase
        self.qkv = torch.nn.Linear(C, 3 * C, bias=False)
        self.proj = torch.nn.Linear(C, C, bias=False)
        self.w12 = torch.nn.Parameter(torch.randn(8 * C, C, device=dev) * C ** -0.5)
        self.w3 = torch.nn.Parameter(torch.randn(C, 4 * C, device=dev) * (4 * C) ** -0.5)

    def forward(self, x):
        C = x.shape[1]
        if self.phase == 2:
            h = fp8_linear(fp8_linear(x, self.qkv.weight)[:, :C], self.proj.weight)
            return fp8_mlp(h, self.w12, self.w3)
        if self.phase == 4:
            h = F.fp8_linear4(x, self.qkv.weight, None, 0, True)
            h = F.fp8_linear4(h[:, :C], self.proj.weight, None, 2, False)
            return F.fp8_mlp4(h, self.w12, self.w3, 4)
        h = F.fp8_linear3(x, self.qkv.weight, None, 0, True)
        h = F.fp8_linear3(h[:, :C], self.proj.weight, None, 1, False)
        return F.fp8_mlp3(h, self.w12, self.w3, 2)


C = 640
blk3 = Blk(C, 3).to(dev)
blk2 = Blk(C, 2).to(dev)
blk2.load_state_dict(blk3.state_dict())
X = [(torch.randn(2048, C, device=dev)).to(BF) for _ in range(3)]   # one distribution: no step outgrows the history
DY = [torch.randn(2048, C, device=dev).to(BF) for _ in range(3)]


def run3(m):
    F.DS.sites, F.DS.st = 8, None
    F.ds_state(dev)
    outs = []
    for x, dy in zip(X, DY):
        m.zero_grad(set_to_none=True)
        y = m(x)
        y.backward(dy)
        outs.append((y.detach().clone(), m.w12.grad.clone(), m.qkv.weight.grad.clone()))
    torch.cuda.synchronize()
    return outs, (F.DS.st.clone() if F.DS.st is not None else None)


def fp32_ref(m, x, dy):
    R = [t.detach().clone().float().requires_grad_() for t in (m.qkv.weight, m.proj.weight, m.w12, m.w3)]
    h = (x.float() @ R[0].t())[:, :C] @ R[1].t()
    a1, a2 = (h @ R[2].t()).chunk(2, -1)
    y = (torch.nn.functional.silu(a1) * a2) @ R[3].t()
    y.backward(dy.float())
    return y, R[2].grad


def compiled(m, cpp):
    torch._dynamo.reset()
    with torch._inductor.config.patch(cpp_wrapper=cpp):
        return run3(torch.compile(m))


p2, _ = run3(blk2)
eager, st_e = run3(blk3)
comp, st_c = compiled(blk3, False)
comp_cpp, st_cc = compiled(blk3, True)   # production compiles with the C++ wrapper (TORCHINDUCTOR_CPP_WRAPPER=1)
same_c = all(torch.equal(a, b) for oe, oc in zip(eager, comp) for a, b in zip(oe, oc)) and torch.equal(st_e, st_c)
same_cc = all(torch.equal(a, b) for oe, oc in zip(eager, comp_cpp) for a, b in zip(oe, oc)) and torch.equal(st_e, st_cc)
same_c = same_c and same_cc
step1 = all(torch.equal(a, b) for a, b in zip(eager[0], p2[0]))   # no history yet: phase 2's arithmetic
errs = []
for k in (1, 2):
    yr, gr = fp32_ref(blk3, X[k], DY[k])
    errs.append((rel(eager[k][0], yr), rel(p2[k][0], yr), rel(eager[k][1], gr), rel(p2[k][1], gr)))
near = all(e3y <= 1.25 * e2y + 5e-3 and e3g <= 1.25 * e2g + 5e-3 for e3y, e2y, e3g, e2g in errs)
calibrated = bool((st_e[:6, F.L_HIST + 1] > 0).all())
good = same_c and step1 and near and calibrated
ok &= bool(good)
print(f"phase 3 block, 3 steps: compiled = eager bit for bit, with and without the C++ wrapper {same_c} ({same_cc} with); "
      f"step 1 (no history) = phase 2 bit for bit {step1}; "
      f"every site calibrated after step 1 {calibrated}; steps 2 and 3 vs fp32 autograd (phase 3 / phase 2): "
      + ", ".join(f"y {e3y:.1e} / {e2y:.1e}, dw12 {e3g:.1e} / {e2g:.1e}" for e3y, e2y, e3g, e2g in errs)
      + ("" if good else "  FAIL"), flush=True)
# 7. phase 4: the casts with transposes against phase 3's, then a block of four GEMMs with FP8 dgrads and wgrads
F.DS.sites, F.DS.st = 8, None
F.ds_state(dev)
x = (torch.randn(4096, 640, device=dev) * 3).to(BF)
q3, s3 = F.dcast(x, F.DS.st, 0, False)
F.ds_reset()
q4, q4t, s4 = F.dcast_t(x, F.DS.st, 0, False, False)
g1 = bits_eq(q4, q3) and bits_eq(q4t, q3.t().contiguous()) and torch.equal(s4, s3)
F.ds_reset()
zq3, zs3 = F.swiglu_dcast(z, F.DS.st, 1)
F.ds_reset()
zq4, zq4t, zs4 = F.swiglu_dcast_t(z, F.DS.st, 1)
g2 = bits_eq(zq4, zq3) and bits_eq(zq4t, zq3.t().contiguous()) and torch.equal(zs4, zs3)
F.ds_reset()
_, dzq3, dzs3 = F.swiglu_bwd_ds(z, da, F.DS.st, 2)
F.ds_reset()
dzq4, dzq4t, dzs4 = F.swiglu_bwd_dt(z, da, F.DS.st, 2)
g3 = (dzq4.dtype != E4 or bits_eq(dzq4, dzq3)) and bits_eq(dzq4t, dzq4.t().contiguous()) and (dzq4.dtype != E4 or torch.equal(dzs4, dzs3))
blk4 = Blk(C, 4).to(dev)
blk4.load_state_dict(blk3.state_dict())
p3, _ = run3(blk3)
e4, st4 = run3(blk4)
c4, st4c = compiled(blk4, True)
same4 = all(torch.equal(a, b) for oe, oc in zip(e4, c4) for a, b in zip(oe, oc)) and torch.equal(st4, st4c)
fwd4 = all(torch.equal(e4[k][0], p3[k][0]) for k in range(3))   # the forward casts and GEMMs are phase 3's
errs4 = []
for k in (1, 2):
    yr, gr = fp32_ref(blk4, X[k], DY[k])
    errs4.append((rel(e4[k][1], gr), rel(p3[k][1], gr)))
near4 = all(e4g <= 2.0 * e3g + 2e-2 for e4g, e3g in errs4)
good = g1 and g2 and g3 and same4 and fwd4 and near4
ok &= bool(good)
print(f"phase 4: dcast_t = dcast and its transpose {g1}; swiglu_dcast_t {g2}; swiglu_bwd_dt (gradients {'e5m2' if F._GRAD_E5 else 'e4m3'}) "
      f"{g3}; block, 3 steps: compiled (C++ wrapper) = eager bit for bit {same4}; forward = phase 3's {fwd4}; dw12 vs fp32 "
      f"(phase 4 FP8 wgrad / phase 3 bf16 wgrad): " + ", ".join(f"{a:.1e} / {b:.1e}" for a, b in errs4) + ("" if good else "  FAIL"),
      flush=True)
print("FP8_NATIVE_TEST " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
