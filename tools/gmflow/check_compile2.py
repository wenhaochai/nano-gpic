#!/usr/bin/env python3
"""CPU check of gm_spread pixel under torch.compile (as net.compile() in production): trainer steps at several text lengths, diagnostics on and off; counts graphs
and graph breaks. Also the order-2 samplers on the compiled model. Run: tools/gmflow/run_vis.sh tools/gmflow/check_compile2.py"""
import torch, torch._dynamo as dynamo
from torch._dynamo.utils import counters
from src.diffusion.base.guidance import simple_guidance_fn
from src.diffusion.flow_matching.sampling_gmflow import GMODESampler, GMSDESampler
from src.diffusion.flow_matching.scheduling import LinearScheduler
from src.diffusion.flow_matching.training_gmflow import GMFlowTrainer
from src.diffusion.flow_matching.training_JiT import constant
from src.models.transformer.JiT_T2I import JiT_T2I
small = dict(input_size=64, in_channels=3, num_groups=4, hidden_size=64, num_blocks=3, num_text_blocks=1, patch_size=16, txt_embed_dim=32, txt_max_length=64, bottleneck_dim=16)
torch.manual_seed(0)
net = JiT_T2I(**small, gm_k=8, gm_spread="pixel")
net.compile()
tr = GMFlowTrainer(scheduler=LinearScheduler(), loss_weight_fn=constant, lognorm_t=True, timeshift=1.0, P_mean=-0.8, P_std=0.8, t_eps=0.05, null_condition_p=0.0, lam=0.5)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
frames = []
for i, L in enumerate((16, 24, 40, 24, 16, 40)):
    tr.diag_now = i % 2 == 0
    x = torch.randn(4, 3, 64, 64).clamp(-1, 1); y = torch.randn(4, L, 32); mask = torch.ones(4, L, dtype=torch.bool); mask[:, L // 2:] = False
    out = tr._impl_trainstep(net, None, None, x, y, {"y_mask": mask})
    opt.zero_grad(); out["loss"].backward(); opt.step()
    frames.append(counters["stats"]["unique_graphs"])
    print(f"step {i} text {L} diag {tr.diag_now}: loss {out['loss'].item():.4f} graphs so far {frames[-1]}" + (f" logsu_pix_std {out['gm/logsu_pix_std'].item():.5f}" if "gm/logsu_pix_std" in out else ""), flush=True)
print("graph breaks:", dict(counters["graph_break"]))
f = lambda x, t, c, **kw: net(x, t, c, y_mask=torch.ones(2, 8, dtype=torch.bool), **kw)
with torch.no_grad():
    for cls, kw in ((GMODESampler, {"substeps": 16}), (GMSDESampler, {})):
        s = cls(scheduler=LinearScheduler(), w_scheduler=LinearScheduler(), guidance_fn=simple_guidance_fn, num_steps=8, guidance=1.0, order=2, rank_seed=False, **kw)
        o = s._impl_sampling(f, torch.randn(2, 3, 64, 64), torch.randn(2, 8, 32), None)[0][-1]
        print(cls.__name__, "order 2 on the compiled pixel model: finite", bool(torch.isfinite(o).all()), "graphs", counters["stats"]["unique_graphs"])
assert frames[-1] == frames[2], f"new graphs after the warm-up: {frames}"
assert not counters["graph_break"], counters["graph_break"]
print("compile check PASS")
