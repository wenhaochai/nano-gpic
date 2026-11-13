"""Port tests of the per-pixel mixture on the final ladder's training path (branch gmflow-ladder; CPU, a vis node).
usage: bash tools/gmflow_ladder/vis.sh tools/gmflow_ladder/test_port.py [--ref <export of 5ac69ab's baselines/PixelGen>]
Each case runs tools/gmflow_ladder/harness.py (the real LightningModel, MuonAdamW, no EMA, warmup + 50% cooldown, checkpoint hooks, CompileWarmup)
in its own process, with torch.compile off (TORCHDYNAMO_DISABLE=1: the compiled path is the GPU smoke's job).
P1  groups: with gm_k = 8 every parameter is in exactly one group; Muon holds only 2D weights of blocks.* and text_refine_blocks.*; the mixture
    head (final_layer.*, gm_logstd.*) is in the AdamW groups; the point model's groups are the mixture's minus gm_logstd.* (same names otherwise).
P2  the mixture trains: finite losses over 8 steps for K = 8 and K = 1, the EMA is off and every checkpoint's EMA slot equals the raw weights,
    the learning rates follow warmup, flat, linear cooldown to 3e-5.
P3  resume: a run stopped at step 4 and resumed to step 8 restores the weights exactly (the probe loss at the start of the resumed launch equals
    the probe loss at the end of the first), keeps the schedule (lrs of steps 5-8 equal the one-piece run's) and ends with finite losses.
P4  (with --ref) the point model on this tree and on 5ac69ab: equal losses at every step, equal groups, equal final weights, bit for bit.
P5  the head's added forward FLOPs equal tools/gmflow_ladder/steps.py's formula (FlopCounterMode on the tiny model, K = 8 and K = 1)."""
import json
import math
import os
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
REF = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else None
TMP = tempfile.mkdtemp(prefix="gm_port_", dir=os.environ.get("TMPDIR", "/scratch/gpfs/GROUP/USER/tmp/cc_tmp"))
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name} {detail}", flush=True)


def run(tag, pix=None, **cfg):
    cfg = dict(dict(out=f"{TMP}/{tag}.pt", root=f"{TMP}/{tag}", k=0, steps=8, seed=0, batch=4, ms_every=2, last_every=4, probe=True), **cfg)   # two CheckpointHooks need different intervals (Lightning's state keys)
    os.makedirs(cfg["root"], exist_ok=True)
    cj = f"{TMP}/{tag}.json"
    json.dump(cfg, open(cj, "w"))
    env = dict(os.environ, TORCHDYNAMO_DISABLE="1", PYTHONPATH=pix or os.path.join(REPO, "baselines", "PixelGen"))
    r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "run.py")] + (["--pix", pix] if pix else []) + ["tools/gmflow_ladder/harness.py", cj],
                       env=env, capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-3000:], flush=True)
        raise SystemExit(f"harness {tag} failed")
    return torch.load(cfg["out"], weights_only=False)


# ---- P1, P2 ----
g8 = run("k8", k=8)
g1 = run("k1", k=1)
p0 = run("point", k=0)
strip = lambda n: n[len("denoiser."):] if n.startswith("denoiser.") else n
for tag, g in (("K=8", g8), ("K=1", g1)):
    names = [strip(n) for grp in g["groups"] for n in grp["names"]]
    check(f"P1 {tag}: every parameter in exactly one group", len(names) == len(set(names)) and set(names) <= set(g["state"]) and len(names) > 20,
          f"({len(names)} parameters in {len(g['groups'])} groups)")
    head = [n for n in names if n.startswith(("final_layer.", "gm_logstd."))]
    check(f"P1 {tag}: the head is in the groups", len([n for n in head if n.startswith('gm_logstd.')]) == 4 and len(head) >= 6, f"({head})")
    check(f"P2 {tag}: finite losses", all(math.isfinite(v) for v in g["losses"]) and len(g["losses"]) == 8, str([round(v, 4) for v in g["losses"]]))
    check(f"P2 {tag}: EMA off, EMA slot equals the raw weights in every checkpoint", g["ema_off"] and g["ema_equal"] and all(g["ema_equal"].values()), str(g["saves"]))
    lr = [row[0] for row in g["lrs"]]
    check(f"P2 {tag}: the rate falls over the cooldown half and ends near 3e-5", all(x >= y for x, y in zip(lr[4:], lr[5:])) and lr[-1] < lr[3] and lr[-1] >= 3e-5 - 1e-12,
          str([f"{v:.2e}" for v in lr]))
pn = sorted(strip(n) for grp in p0["groups"] for n in grp["names"])
mn = sorted(strip(n) for grp in g8["groups"] for n in grp["names"] if not strip(n).startswith("gm_logstd."))
check("P1 the point model's parameter names = the mixture's without gm_logstd", pn == mn, f"({len(pn)} against {len(mn)})")
check("P1 the same lrs in the point model and the mixture", p0["lrs"] == g8["lrs"] == g1["lrs"])

# ---- P3 ----
a = run("k8_first", k=8, steps=4, total_steps=8)
b = run("k8_resumed", k=8, steps=8, total_steps=8, root=f"{TMP}/k8_first", resume=f"{TMP}/k8_first/last.ckpt")
check("P3 the resumed launch starts at step 4 with the first launch's weights", b["probe_start"][0] == 4 and a["probe_end"][0] == 4 and b["probe_start"][1] == a["probe_end"][1],
      f"(probe {a['probe_end']} against {b['probe_start']})")
check("P3 the schedule continues", b["lrs"] == g8["lrs"][4:], f"({[f'{r[0]:.2e}' for r in b['lrs']]})")
check("P3 the resumed losses are finite", len(b["losses"]) == 4 and all(math.isfinite(v) for v in b["losses"]), str([round(v, 4) for v in b["losses"]]))

# ---- P4 ----
if REF:
    r0 = run("point_ref", pix=REF, k=0)
    check("P4 the point model: losses equal 5ac69ab's", r0["losses"] == p0["losses"], f"({[round(v, 5) for v in p0['losses']]})")
    check("P4 the point model: groups equal 5ac69ab's", r0["groups"] == p0["groups"])
    check("P4 the point model: final weights equal 5ac69ab's", r0["state"].keys() == p0["state"].keys() and all(torch.equal(v, p0["state"][k]) for k, v in r0["state"].items()))
else:
    print("SKIP P4 (no --ref)")

# ---- P5 and the Muon marks ----
sys.path.insert(0, os.path.join(REPO, "baselines", "PixelGen"))
from torch.utils.flop_counter import FlopCounterMode   # noqa: E402
from src.models.transformer.JiT_T2I import JiT_T2I   # noqa: E402
from src.optim.muon import tag_muon_params   # noqa: E402
SMALL = dict(input_size=64, in_channels=3, num_groups=4, hidden_size=64, num_blocks=3, num_text_blocks=1, patch_size=16, txt_embed_dim=32, txt_max_length=32, bottleneck_dim=16)
net = JiT_T2I(**SMALL, gm_k=8)
tag_muon_params(net)
tagged = [n for n, q in net.named_parameters() if hasattr(q, "_muon_parts")]
check("P1 Muon marks only block matrices, none of the head", len(tagged) > 5 and all(n.startswith(("blocks.", "text_refine_blocks.")) for n in tagged)
      and all(q.ndim == 2 for n, q in net.named_parameters() if hasattr(q, "_muon_parts")), f"({len(tagged)} marked)")
ref = JiT_T2I(**SMALL)
tag_muon_params(ref)
check("P1 the same Muon marks as the point model", tagged == [n for n, q in ref.named_parameters() if hasattr(q, "_muon_parts")])


def flops(k, mode):
    torch.manual_seed(0)
    m = JiT_T2I(**SMALL, **({"gm_k": k} if k else {})).eval()
    x, t, y = torch.randn(1, 3, 64, 64), torch.full((1,), 0.5), torch.randn(1, 32, 32)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(x, t, y, **({"gm_mode": mode} if k else {}))
    return fc.get_total_flops()


d, L, C, pp = 64, (64 // 16) ** 2, 3, 16 * 16
f0 = flops(0, None)
for k in (8, 1):
    added = flops(k, "full") - f0   # FlopCounterMode counts two per multiply-add, as the ladder's FLOPs do (C = 6ND)
    want = 2 * L * d * (k * (C + 1) * pp - C * pp) + 2 * (d * 1024 + 1024)
    check(f"P5 K={k}: added forward FLOPs = steps.py's formula", added == want, f"(counted {added}, formula {want})")
print("ALL PASS" if ok else "SOME FAILED", flush=True)
sys.exit(0 if ok else 1)
