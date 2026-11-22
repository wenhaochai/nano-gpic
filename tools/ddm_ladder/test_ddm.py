"""CPU tests of the distributional denoiser (docs/ddm.md). Run on a pilot CPU: PYTHONPATH=baselines/PixelGen python -I tools/ddm_ladder/test_ddm.py
  1. the point model and the old implicit sampler (xi_mode "add") are unchanged: same weights from the same seed, same outputs, as the code of DDM_BASE;
  2. xi_mode "cat": widths, the gate at 1, the m samples equal at init (the blocks start as the identity, the readout drops the noise channels);
  3. the widened blocks' wiring: a cat model holding a point model's weights in its first d channels, zeros in the rest and the norm weights scaled by
     sqrt(d / (d + c)), gives the point model's output for xi = 0;
  4. the samples depend on xi once the blocks are not the identity, and the gate gets a gradient;
  5. the score: kernel_patch = image size equals the whole-image score; the local score equals a loop over patches; the linear schedules equal a loop
     over images; with lam 0 and beta 2 the score is each sample's squared error;
  6. Muon tags the widened matrices with their fused parts;
  7. the FLOP ratio of tools/ddm_ladder/steps.py against torch's FlopCounterMode on the d10 shape (768 x (10 + 2), 256 tokens, a 56-token caption).
     FlopCounterMode does not count SDPA's products on a CPU (its point model is fwd() less them), so the check leaves them out of both sides."""
import importlib.util
import math
import os
import subprocess
import sys
import tempfile

import torch

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WT, "baselines", "PixelGen"))
sys.path.insert(0, os.path.join(WT, "tools", "ddm_ladder"))
from src.models.transformer.JiT_T2I import JiT_T2I   # noqa: E402

torch.set_grad_enabled(False)
KW = dict(input_size=64, in_channels=3, num_groups=2, hidden_size=128, num_blocks=4, num_text_blocks=1, patch_size=16, txt_embed_dim=64,
          txt_max_length=16, bottleneck_dim=32)
FAILS = []


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def old_module():
    base = os.environ.get("DDM_BASE", "14fa39c")
    code = subprocess.check_output(["git", "-C", WT, "show", f"{base}:baselines/PixelGen/src/models/transformer/JiT_T2I.py"], text=True)
    p = os.path.join(tempfile.mkdtemp(), "JiT_T2I_old.py")
    open(p, "w").write(code)
    spec = importlib.util.spec_from_file_location("JiT_T2I_old", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def inputs(B=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, 3, 64, 64, generator=g)
    t = torch.rand(B, generator=g)
    y = torch.randn(B, KW["txt_max_length"], KW["txt_embed_dim"], generator=g)
    return x, t, y


def same_weights(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(sa[k].shape == sb[k].shape and torch.equal(sa[k], sb[k]) for k in sa)


def test_unchanged():
    old = old_module()
    x, t, y = inputs()
    for extra in (dict(), dict(xi_dim=16, xi_block=2)):
        torch.manual_seed(0); a = old.JiT_T2I(**KW, **extra).eval()
        torch.manual_seed(0); b = JiT_T2I(**KW, **extra).eval()
        ok(same_weights(a, b), f"same weights from the same seed as the old code {extra}")
        for n, p in a.named_parameters():   # non-trivial outputs: the zero-initialized layers get random weights, the same in both
            q = dict(b.named_parameters())[n]
            if p.abs().sum() == 0:
                w = torch.randn(p.shape, generator=torch.Generator().manual_seed(hash(n) % 2 ** 31)) * 0.02
                p.copy_(w); q.copy_(w)
        m = 2 if extra else 1
        torch.manual_seed(5); oa = a(x, t, y, m=m)
        torch.manual_seed(5); ob = b(x, t, y, m=m)
        ok(torch.equal(oa, ob), f"same outputs as the old code {extra} (max diff {(oa - ob).abs().max():.3g})")


def cat_model(c=32, block=2, seed=0):
    torch.manual_seed(seed)
    return JiT_T2I(**KW, xi_dim=c, xi_block=block, xi_mode="cat").eval()


def test_cat_init():
    d, c = KW["hidden_size"], 32
    q = cat_model(c)
    widths = [blk.norm1.weight.shape[0] for blk in q.blocks]
    ok(widths == [d, d, d + c, d + c], f"block widths {widths}")
    ok(q.xi_gate.weight.abs().sum() == 0 and q.xi_gate.bias.abs().sum() == 0, "the gate starts at 1")
    blk = q.blocks[3]
    shapes = (blk.attn.qkv_x.weight.shape, blk.attn.kv_y.weight.shape, blk.attn.proj.weight.shape, blk.mlp.w12.weight.shape, blk.mlp.w3.weight.shape,
              blk.adaLN_modulation[0].weight.shape)
    want = ((3 * d, d + c), (2 * d, d), (d + c, d), (8 * d, d + c), (d + c, 4 * d), (6 * (d + c), d))
    ok(shapes == tuple(torch.Size(w) for w in want), f"widened block shapes {shapes}")
    x, t, y = inputs()
    out = q(x, t, y, m=4)
    ok(out.shape == (12, 3, 64, 64), f"output shape {tuple(out.shape)}")
    o = out.view(3, 4, -1)
    ok(torch.equal(o, o[:, :1].expand_as(o)), "the 4 samples are equal at init")


def test_wiring():
    d, c, blk0 = KW["hidden_size"], 32, 2
    torch.manual_seed(0)
    p = JiT_T2I(**KW).eval()
    for n, w in p.named_parameters():   # every weight random, so the test sees every path
        w.copy_(torch.randn(w.shape, generator=torch.Generator().manual_seed(hash(n) % 2 ** 31)) * (0.05 if w.ndim > 1 else 0.3) + (1.0 if n.endswith("norm1.weight") or n.endswith("norm2.weight") else 0.0))
    q = cat_model(c, blk0)
    sp, sq = p.state_dict(), q.state_dict()
    D = d + c
    for k, v in sq.items():
        if k not in sp:
            continue
        w = sp[k]
        if w.shape == v.shape:
            v.copy_(w)
            continue
        v.zero_()
        if k.endswith("norm1.weight") or k.endswith("norm2.weight"):
            v[:d] = w * math.sqrt(d / D)
        elif k.endswith("attn.qkv_x.weight") or k.endswith("mlp.w12.weight"):
            v[:, :d] = w
        elif k.endswith("attn.proj.weight") or k.endswith("mlp.w3.weight") or k.endswith("attn.proj.bias"):
            v[:d] = w
        elif k.endswith("adaLN_modulation.0.weight") or k.endswith("adaLN_modulation.0.bias"):
            for j in range(6):
                v[j * D:j * D + d] = w[j * d:(j + 1) * d]
        else:
            raise AssertionError(f"unexpected shape change: {k} {tuple(w.shape)} -> {tuple(v.shape)}")
    missing = [k for k in sq if k not in sp]
    ok(missing == ["xi_gate.weight", "xi_gate.bias"], f"only the gate is new: {missing}")
    x, t, y = inputs()
    ref = p(x, t, y)
    xi = torch.zeros(3 * 4, (64 // 16) ** 2, c)
    out = q(x, t, y, m=4, xi=xi).view(3, 4, 3, 64, 64)
    err = (out - ref[:, None]).abs().max().item() / ref.abs().max().item()
    ok(err < 1e-4, f"the widened blocks with zero extra weights reproduce the point model (relative max error {err:.2e})")


def test_xi_and_gate():
    q = cat_model(32)
    for n, w in q.named_parameters():
        if "adaLN_modulation" in n or n.startswith("final_layer.linear") or n.startswith("xi_gate"):
            w.copy_(torch.randn(w.shape, generator=torch.Generator().manual_seed(hash(n) % 2 ** 31)) * 0.05)
    x, t, y = inputs()
    a, b = q(x, t, y, m=4).view(3, 4, -1), None
    ok((a[:, 0] - a[:, 1]).abs().max() > 1e-4, "the samples differ once the blocks are not the identity")
    xi = torch.randn(12, 16, 32, generator=torch.Generator().manual_seed(3))
    ok(torch.equal(q(x, t, y, m=4, xi=xi), q(x, t, y, m=4, xi=xi)), "a given xi gives the same samples")
    with torch.enable_grad():
        q.zero_grad()
        q(x, t, y, m=4, xi=xi).square().mean().backward()
    gn = q.xi_gate.weight.grad.abs().sum().item()
    ok(gn > 0, f"the gate gets a gradient ({gn:.3g})")


def test_score():
    from src.diffusion.flow_matching.scheduling import LinearScheduler
    from src.diffusion.flow_matching.training_JiT import constant
    from src.diffusion.flow_matching.training_implicit import ImplicitJiTTrainer
    mk = lambda **k: ImplicitJiTTrainer(scheduler=LinearScheduler(), loss_weight_fn=constant, lognorm_t=True, timeshift=1.0, P_mean=-0.8, P_std=0.8,
                                        t_eps=0.05, null_condition_p=0.0, samples=4, **k)
    gen = torch.Generator().manual_seed(7)
    B, m = 5, 4
    x = torch.randn(B, 3, 64, 64, generator=gen)
    g = x[:, None] + 0.3 * torch.randn(B, m, 3, 64, 64, generator=gen)
    t = torch.tensor([0.0, 0.2, 0.5, 0.8, 0.97])
    w = 1 / (1 - t).clamp_min(0.05)
    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]

    def manual(beta_of, lam_of, p):
        out = []
        for b in range(B):
            be, la = beta_of(t[b]), lam_of(t[b])
            def np_(d):   # mean over patches of (mean square + 1e-8)^(beta / 2)
                d = d * w[b]
                if not p:
                    return (d.square().mean() + 1e-8) ** (be / 2)
                vals = [(d[:, r:r + p, s:s + p].square().mean() + 1e-8) ** (be / 2) for r in range(0, 64, p) for s in range(0, 64, p)]
                return torch.stack(vals).mean()
            td = torch.stack([np_(g[b, i] - x[b]) for i in range(m)]).mean()
            bt = torch.stack([np_(g[b, i] - g[b, j]) for i, j in pairs]).mean()
            out.append(td - 0.5 * la * bt)
        return torch.stack(out)

    s_glob, _, _ = mk(beta=1.0, lam=1.0).score(g, x, t)
    s_one, _, _ = mk(beta=1.0, lam=1.0, kernel_patch=64).score(g, x, t)
    ok(torch.allclose(s_glob, s_one, rtol=1e-5, atol=1e-6), "one patch of the whole image equals the whole-image score")
    ok(torch.allclose(s_glob, manual(lambda _: 1.0, lambda _: 1.0, 0), rtol=1e-5, atol=1e-6), "the whole-image score equals the loop")
    s_loc, _, _ = mk(beta=1.0, lam=1.0, kernel_patch=16).score(g, x, t)
    ok(torch.allclose(s_loc, manual(lambda _: 1.0, lambda _: 1.0, 16), rtol=1e-5, atol=1e-6), "the local score equals the loop over patches")
    s_lin, _, _ = mk(beta=0.1, lam=1.0, kernel_patch=16, schedule="linear").score(g, x, t)
    ref = manual(lambda tt: 2 - 1.9 * (1 - tt), lambda tt: 1.0 * (1 - tt), 16)
    ok(torch.allclose(s_lin, ref, rtol=1e-5, atol=1e-6), f"the linear schedules equal the loop (max diff {(s_lin - ref).abs().max():.3g})")
    s_lin1, _, _ = mk(beta=0.1, lam=1.0, kernel_patch=16, schedule="linear").score(g, x, torch.ones(B))
    mse = ((g - x[:, None]) / 0.05).square().view(B, m, 3, 4, 16, 4, 16).mean((2, 4, 6)).add(1e-8).mean((1, 2, 3))
    ok(torch.allclose(s_lin1, mse, rtol=1e-5), "at t = 1 the score is each sample's squared error (lam 0, beta 2)")


def test_muon_tags():
    from src.optim.muon import tag_muon_params
    q = cat_model(32)
    tag_muon_params(q)
    blk = q.blocks[3]
    parts = {n: getattr(p, "_muon_parts", None) for n, p in blk.named_parameters() if p.ndim == 2}
    want = {"attn.qkv_x.weight": 3, "attn.kv_y.weight": 2, "attn.proj.weight": 1, "mlp.w12.weight": 2, "mlp.w3.weight": 1, "adaLN_modulation.0.weight": 6}
    ok(parts == want, f"Muon parts of a widened block {parts}")
    ok(getattr(q.xi_gate.weight, "_muon_parts", None) is None, "the gate stays with AdamW")


def test_flops():
    from torch.utils.flop_counter import FlopCounterMode
    import steps
    kw = dict(input_size=256, in_channels=3, num_groups=12, hidden_size=768, num_blocks=10, num_text_blocks=2, patch_size=16, txt_embed_dim=2048,
              txt_max_length=320, bottleneck_dim=128)
    T = 56
    y = torch.randn(1, T, 2048)
    y_mask = torch.ones(1, T, dtype=torch.bool)
    x, t = torch.randn(1, 3, 256, 256), torch.rand(1)
    fl = {}
    for name, extra, m in (("point", dict(), 1), ("ddm", dict(xi_dim=256, xi_block=8, xi_mode="cat"), 4)):
        net = JiT_T2I(**kw, **extra).eval()
        with FlopCounterMode(display=False) as fc:
            net(x, t, y, y_mask=y_mask, m=m)
        fl[name] = fc.get_total_flops()
    r_fc = fl["ddm"] / fl["point"] - 1
    r_an = steps.ratio(768, 10, 2, T=T, T2=T * T)
    r_na = steps.added(768, T, attn=False) / steps.fwd_no_attn(768, 10, 2, T, T * T)   # FlopCounterMode does not count SDPA on a CPU
    print(f"  FlopCounterMode: point {fl['point'] / 1e9:.3f} GFLOP, ddm {fl['ddm'] / 1e9:.3f} GFLOP, r {r_fc:.4f}; steps.py r {r_an:.4f}, without the "
          f"attention products {r_na:.4f}; fwd() point {steps.fwd(768, 10, 2, T, T * T) / 1e9:.3f} GFLOP, without them "
          f"{steps.fwd_no_attn(768, 10, 2, T, T * T) / 1e9:.3f}", flush=True)
    ok(abs(steps.fwd_no_attn(768, 10, 2, T, T * T) / fl["point"] - 1) < 0.01, "fwd() without the attention products within 1% of FlopCounterMode's point model")
    ok(abs(r_na - r_fc) / r_fc < 0.01, f"steps.py's FLOP ratio without the attention products within 1% of FlopCounterMode's ({r_na:.4f} vs {r_fc:.4f})")


if __name__ == "__main__":
    for f in (test_unchanged, test_cat_init, test_wiring, test_xi_and_gate, test_score, test_muon_tags, test_flops):
        print(f"== {f.__name__}", flush=True)
        try:
            f()
        except Exception as e:   # noqa: BLE001
            import traceback
            traceback.print_exc()
            ok(False, f"{f.__name__} raised {type(e).__name__}: {e}")
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
