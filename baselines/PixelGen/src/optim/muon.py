"""Muon for the hidden matrices of the transformer blocks, AdamW (official settings) for every other parameter.

Muon update of a matrix W, torch Linear layout [d_out, d_in] (Moonlight scaling, Liu et al. 2025, arXiv 2502.16982):
    m <- beta m + (1 - beta) g                         momentum (beta 0.95)
    u <- (1 - beta) g + beta m                         Nesterov
    O <- msign(u)                                      5 Newton-Schulz steps in bf16 (Keller Jordan's coefficients)
    W <- (1 - lr wd) W - lr * 0.2 sqrt(max(d_out, d_in)) * O
msign(u) of an [a, b] matrix has RMS 1/sqrt(max(a, b)), so the update has RMS 0.2 x lr, the size of a typical AdamW
update, and AdamW's lr and weight decay carry over (kexue.fm/archives/11416). The momentum scale does not matter:
msign is invariant to it.

split=True (CMuon, arXiv 2608.02502): a fused matrix is cut along d_out into its parts (q, k, v; k, v; gate, up; the
six adaLN modulation vectors) and each part is orthogonalized and scaled on its own. tag_muon_params marks the
matrices and their parts.

Distributed (DDP: every rank holds the same averaged gradient and the same momentum): the Newton-Schulz work is split
over ranks. Each matrix part ("unit") has one owner rank, chosen by a deterministic longest-processing-time assignment;
the owner writes O (bf16) into its slot of a flat buffer; one all_gather_into_tensor gives every rank
every update; every rank applies them all, so the weights stay identical across ranks.
"""
import math

import torch
import torch.distributed as dist
from torch.optim.adam import adam as _adam_fn

NS_COEFFS = (3.4445, -4.7750, 2.0315)
FUSED_PARTS = {"attn.qkv_x.weight": 3, "attn.qkv.weight": 3, "attn.kv_y.weight": 2, "mlp.w12.weight": 2,
               "adaLN_modulation.0.weight": 6}


def tag_muon_params(model, prefixes=("blocks.", "text_refine_blocks.")):
    """Mark the 2D weights inside the transformer blocks for Muon: p._muon_parts = number of fused parts along d_out.
    Everything outside the blocks (patch, time and text embeddings, the final layer) and every non-matrix parameter
    stays with AdamW. Returns (n_tagged, numel_tagged)."""
    n = numel = 0
    for name, p in model.named_parameters():
        if p.ndim != 2 or not p.requires_grad or not name.startswith(prefixes):
            continue
        parts = next((k for suffix, k in FUSED_PARTS.items() if name.endswith(suffix)), 1)
        assert p.shape[0] % parts == 0, (name, tuple(p.shape), parts)
        p._muon_parts = parts
        p._muon_name = name
        n += 1
        numel += p.numel()
    return n, numel


@torch.no_grad()
def msign(G, steps=5, dtype=torch.bfloat16):
    """Orthogonalize each matrix of G ([..., a, b]) with the quintic Newton-Schulz iteration (Keller Jordan), in bf16 by
    default. In bf16 the result moves by up to ~5% of its largest entry with the GEMM algorithm (e.g. batch size), so
    exact comparisons use dtype=torch.float32."""
    a, b, c = NS_COEFFS
    X = G.to(dtype)
    tall = G.size(-2) > G.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.mT if tall else X


class MuonAdamW(torch.optim.Optimizer):
    """One optimizer for Lightning's automatic optimization: Muon for parameters tagged by tag_muon_params, AdamW for
    the rest. lr, betas, eps, weight_decay: the AdamW settings (also those of each incoming param group). muon_lr and
    muon_weight_decay: Muon's (default: the AdamW values, as the Moonlight scaling intends)."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, muon_lr=None,
                 muon_weight_decay=None, muon_momentum=0.95, muon_nesterov=True, ns_steps=5, split=False,
                 fused_adamw=True, ns_dtype="bfloat16"):
        groups = list(params)
        if groups and not isinstance(groups[0], dict):
            groups = [{"params": groups}]
        adam_groups, muon_params = [], []
        for g in groups:
            ps = list(g["params"])
            muon_params += [p for p in ps if hasattr(p, "_muon_parts")]
            rest = [p for p in ps if not hasattr(p, "_muon_parts")]
            if rest:
                adam_groups.append({**g, "params": rest, "use_muon": False})
        all_groups = adam_groups
        if muon_params:
            all_groups = all_groups + [{
                "params": muon_params, "use_muon": True,
                "lr": muon_lr if muon_lr is not None else lr,
                "weight_decay": muon_weight_decay if muon_weight_decay is not None else weight_decay,
                "momentum": muon_momentum, "nesterov": muon_nesterov, "ns_steps": ns_steps, "split": split}]
        defaults = dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay, use_muon=False)
        super().__init__(all_groups, defaults)
        self.fused_adamw = fused_adamw
        self.ns_dtype = getattr(torch, ns_dtype)
        self._plan = None
        if muon_params and (not dist.is_initialized() or dist.get_rank() == 0):
            n_adam = sum(p.numel() for g in adam_groups for p in g["params"])
            n_muon = sum(p.numel() for p in muon_params)
            print(f"[MuonAdamW] Muon: {len(muon_params)} matrices, {n_muon / 1e6:.1f}M params (split={split}); "
                  f"AdamW: {sum(len(g['params']) for g in adam_groups)} tensors, {n_adam / 1e6:.1f}M params", flush=True)

    # ---------------------------------------------------------------------------------------------------------------
    def _build_plan(self, group):
        """Units (matrix parts), their owners and offsets in the flat all-gather buffer. Deterministic on every rank."""
        world = dist.get_world_size() if dist.is_initialized() else 1
        units = []
        for i, p in enumerate(group["params"]):
            parts = p._muon_parts if group["split"] else 1
            rows = p.shape[0] // parts
            for k in range(parts):
                units.append({"p": i, "r0": k * rows, "r1": (k + 1) * rows, "shape": (rows, p.shape[1])})
        load = [0.0] * world
        order = sorted(range(len(units)), key=lambda j: (-math.prod(units[j]["shape"]) * min(units[j]["shape"]), j))
        for j in order:   # longest processing time first: Newton-Schulz cost ~ rows * cols * min(rows, cols)
            r = min(range(world), key=lambda q: (load[q], q))
            units[j]["owner"] = r
            load[r] += math.prod(units[j]["shape"]) * min(units[j]["shape"])
        offsets = [0] * world
        for u in units:
            u["off"] = offsets[u["owner"]]
            offsets[u["owner"]] += math.prod(u["shape"])
        shard = max(offsets) if offsets else 0
        rank = dist.get_rank() if dist.is_initialized() else 0
        mine = {}   # shape -> list of own units, so Newton-Schulz runs batched per shape
        for u in units:
            if u["owner"] == rank:
                mine.setdefault(u["shape"], []).append(u)
        return {"world": world, "rank": rank, "units": units, "shard": shard, "mine": mine}

    @torch.no_grad()
    def _muon_step(self, group):
        params = group["params"]
        grads, bufs = [], []
        for p in params:
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            st = self.state[p]
            if "momentum_buffer" not in st:
                st["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            grads.append(g)
            bufs.append(st["momentum_buffer"])
        beta, lr, wd = group["momentum"], group["lr"], group["weight_decay"]
        torch._foreach_lerp_(bufs, grads, 1 - beta)                      # m <- beta m + (1 - beta) g
        if self._plan is None:
            self._plan = self._build_plan(group)
        plan = self._plan
        dev = params[0].device
        shard = torch.zeros(plan["shard"], dtype=self.ns_dtype, device=dev)
        for shape, us in plan["mine"].items():
            ins = []
            for u in us:
                g, m = grads[u["p"]][u["r0"]:u["r1"]], bufs[u["p"]][u["r0"]:u["r1"]]
                ins.append(g.lerp(m, beta) if group["nesterov"] else m)   # (1 - beta) g + beta m
            O = msign(torch.stack(ins), steps=group["ns_steps"], dtype=self.ns_dtype)
            for u, o in zip(us, O):
                shard[u["off"]:u["off"] + o.numel()].copy_(o.flatten())
        if plan["world"] > 1:
            full = torch.empty(plan["world"] * plan["shard"], dtype=self.ns_dtype, device=dev)
            dist.all_gather_into_tensor(full, shard)
        else:
            full = shard
        if wd:
            torch._foreach_mul_(params, 1 - lr * wd)
        for u in plan["units"]:   # W <- W - lr * 0.2 sqrt(max(d_out, d_in)) * O, the bf16 O promoted to fp32 in the add
            o0 = u["owner"] * plan["shard"] + u["off"]
            upd = full[o0:o0 + math.prod(u["shape"])].view(u["shape"])
            params[u["p"]][u["r0"]:u["r1"]].add_(upd, alpha=-lr * 0.2 * math.sqrt(max(u["shape"])))

    @torch.no_grad()
    def _adamw_step(self, group):
        params, grads, m1, m2, steps = [], [], [], [], []
        for p in group["params"]:
            if p.grad is None:
                continue
            st = self.state[p]
            if not st:
                st["step"] = (torch.zeros((), dtype=torch.float32, device=p.device) if self.fused_adamw
                              else torch.tensor(0.0, dtype=torch.float32))
                st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            params.append(p); grads.append(p.grad); m1.append(st["exp_avg"]); m2.append(st["exp_avg_sq"])
            steps.append(st["step"])
        if not params:
            return
        b1, b2 = group["betas"]
        # torch >= 2.6: AdamW is Adam with decoupled weight decay (the same call torch.optim.AdamW makes)
        _adam_fn(params, grads, m1, m2, [], steps, foreach=None if self.fused_adamw else True, capturable=False,
                 differentiable=False, fused=self.fused_adamw or None, grad_scale=None, found_inf=None,
                 has_complex=False, decoupled_weight_decay=True, amsgrad=False, beta1=b1, beta2=b2, lr=group["lr"],
                 weight_decay=group["weight_decay"], eps=group["eps"], maximize=False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group.get("use_muon"):
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss
