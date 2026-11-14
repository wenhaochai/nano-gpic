"""Mixture-of-experts feed-forward for the image blocks (branch moe; plan tmp/ladder_prof/moe/PLAN.md, owner 2026-11-13).

A dense SwiGLU of hidden H (JiT_T2I: H = 4 d) becomes n experts of hidden h. A token uses G = H / h of them (exactly G under
token choice, G on average under expert choice), so the active FLOPs equal the dense layer's, plus the router (d x n per
token). The plan's notation: expansion X = n h / H (all expert parameters over the dense layer's), granularity G = H / h,
n = G X. With n = 1, G = 1 the layer is the dense SwiGLU (tools/moe/tests/test_moe.py).

Parameters are 2-D with the experts stacked along the rows, so FSDP2's Shard(0) and Muon's row units need no change:
  e12 [n * 2h, d]   expert e = rows [e * 2h, (e + 1) * 2h): the gate half, then the up half (SwiGLU's w12, Linear layout)
  e3t [n * h, d]    expert e = rows [e * h, (e + 1) * h): its output matrix transposed (w3.T), y = act @ e3t[e]
  router.weight [n, d]   (learned routes only; AdamW, fp32, kept out of FSDP2's bf16 sharding: muon_units, fsdp_keep)
Initialization: the uniform bounds of the dense matrices (xavier on [2H, d] and [d, H]), so the G active experts of a token
have the statistics of a dense layer at the start. Muon (src/optim/muon.py, MuP scale per row unit): e12's units take
sqrt(h / d); e3t's take sqrt(d / h) / G, because the G expert updates of one token add up: the stacked update of the active
experts then has the spectral scale of the dense layer's (sqrt(H / d) and sqrt(d / H)).

Routes (route=):
  tc     token choice: each token takes its top-G experts by sigmoid(score) + bias. The bias is not trained: after every
         optimizer step the callback moves it by rate * sign(mean load - load) (DeepSeek's loss-free balancing, 2408.15664).
         Static shapes need a capacity: each expert takes at most capacity * G * tokens / n tokens in training (the
         surplus is dropped and counted); in evaluation nothing is dropped.
  ecb    expert choice over the micro-batch: each expert takes exactly G * tokens / n tokens, those with its highest
         scores (torch.topk along the tokens). A token gets 0 to n experts. The boundary score per expert is averaged (EMA, in the callback) into moe_thr; in evaluation a
         token takes expert e when sigmoid(score) >= moe_thr[e], so one image's routing does not depend on the batch.
  eci    expert choice inside each image: each expert takes exactly G * T / n tokens of every image (EC-DiT's rule).
         The same in training and evaluation.
  noise  no router: all tokens of an image go to the expert of its noise-level bin (t_edges: n - 1 bin edges of t). G = 1.
  pos    no router: a fixed partition of the token grid into n tiles. G = 1.
  hash   no router: a fixed random partition of the token positions into n equal sets. G = 1.
Gates (learned routes): expert e's output for token i is weighted G * sigmoid(s_ie) / sum of sigmoid(s_ij) over the token's
selected experts, so the weights of a token sum to G (one dense layer's worth) whatever the number of experts it got.
router_in: "mod" = the layer's input (normalized and modulated by the timestep); "plain" = the normalized token before the
modulation plus the timestep embedding (Nucleus-Image's decoupled routing, 2604.12163).

Statistics (training only) go to the non-persistent buffer moe_stat = [load per expert (n), boundary score per expert (n),
dropped assignments, tokens, calls]; src/callbacks/moe_balance.py sums it over ranks, updates moe_bias / moe_thr and logs.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

ROUTES = ("tc", "ecb", "eci", "noise", "pos", "hash")
# GPIC_MOE_KERNELS=1: the gather's backward and the combine as one deterministic segmented sum per token (src/models/layers/moe_ops.py,
# Triton; the same values up to the order of a token's fp32 terms). Off: the torch path below (fp32 atomics).
# GPIC_MOE_KERNELS=2 (2026-11-14, after the step profile w2686): the same two sums in token order through the inverse slot map (moe_ops.py v2:
# no sort, no atomics, no fp32 buffer, deterministic), and the gather without the fp32 round trip.
_KERNELS = int(os.environ.get("GPIC_MOE_KERNELS", "0"))
# GPIC_MOE_BMM_WT=1 (2026-11-14, w2722's profile): the experts' first GEMM, xe @ W^T per expert, with its gradients in their own layouts.
# autograd of bmm(xe, W.mT) makes W's gradient as [n, d, 2h] and copies it into the parameter's [n, 2h, d] (triton_poi_fused_clone,
# 0.09 ms a layer at d12 with 16 experts); here the backward computes bmm(g, W) and bmm(g.mT, xe), both contiguous. The same products;
# cuBLAS may pick another kernel for the transposed problem, so the bf16 rounding of the gradient may differ.
_BMM_WT = os.environ.get("GPIC_MOE_BMM_WT", "0") == "1"
# GPIC_MOE_KERNELS=4 (2026-11-14, w2722's profile and the granularity runs): v3 with the gate on the experts' hidden activations, not on
# their outputs (moe_ops.py v4): the combine is a plain sum over a token's slots and its backward a plain row gather, and the experts'
# outputs [slots, d] are not kept for the backward. The same function; the forward differs from v3's in bf16 rounding (the gated hidden is
# rounded before the last GEMM).
# GPIC_MOE_ROUTER=1 (2026-11-14, w2722's profile): the router's scores from the layer's input in its own dtype (moe_ops.py
# gpic::moe_scores: IEEE fp32 products and sums, no fp32 copy of the input, the input's gradient written in its dtype). Off: F.linear on
# an fp32 cast of the input (a TF32 GEMM on an H100). The scores differ in fp32 rounding.
_ROUTER = os.environ.get("GPIC_MOE_ROUTER", "0") == "1"


class _BmmWT(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, xe, w):   # xe [n, cap, d], w [n, k, d] -> [n, cap, k]
        ctx.save_for_backward(xe, w)
        return torch.bmm(xe, w.mT)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")   # the backward under the forward's autocast state (bf16 GEMMs on fp32 weights)
    def backward(ctx, g):
        xe, w = ctx.saved_tensors
        return torch.bmm(g, w), torch.bmm(g.mT, xe)


def _bmm_wt(xe, w):
    return _BmmWT.apply(xe, w) if _BMM_WT else torch.bmm(xe, w.mT)


class MoEFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, experts, active=1, route="tc", capacity=1.5, shared_hidden=0, router_in="mod",
                 t_edges=None, tokens=256, bias_rate=1e-3, lr_mult=1.0):
        super().__init__()
        assert route in ROUTES, route
        assert router_in in ("mod", "plain"), router_in
        assert experts >= 1 and active >= 1 and (hidden_dim - shared_hidden) % active == 0, (hidden_dim, shared_hidden, active)
        self.dim, self.hidden_dim, self.n, self.G = dim, hidden_dim, experts, active
        self.h = (hidden_dim - shared_hidden) // active
        self.route, self.capacity, self.router_in, self.bias_rate = route, float(capacity), router_in, float(bias_rate)
        self.lr_mult = float(lr_mult)   # factor on the experts' Muon rate (the plan's arm H)
        self.learned = route in ("tc", "ecb", "eci")
        assert self.learned or active == 1, "a fixed route gives each token one expert"
        assert experts >= active or route != "tc", (experts, active)
        n, h = self.n, self.h
        self.e12 = nn.Parameter(torch.empty(n * 2 * h, dim))
        self.e3t = nn.Parameter(torch.empty(n * h, dim))
        self.router = nn.Linear(dim, n, bias=False) if self.learned else None
        if shared_hidden:
            from src.models.layers.swiglu import SwiGLU
            self.shared = SwiGLU(dim, shared_hidden)
        else:
            self.shared = None
        self.register_buffer("moe_bias", torch.zeros(n))                    # tc: added to sigmoid(score) for the selection
        self.register_buffer("moe_thr", torch.full((n,), 0.5))              # ecb: evaluation thresholds on sigmoid(score)
        self.register_buffer("moe_thr_set", torch.zeros((), dtype=torch.bool))   # ecb: moe_thr holds a measured value
        self.register_buffer("moe_stat", torch.zeros(2 * n + 3), persistent=False)
        if route == "noise":
            assert t_edges is not None and len(t_edges) == n - 1, "noise route: n - 1 bin edges of t"
            self.register_buffer("moe_edges", torch.tensor([float(v) for v in t_edges], dtype=torch.float32))
        if route in ("pos", "hash"):
            assert tokens % n == 0, (tokens, n)
            side = int(round(math.sqrt(tokens)))
            assert side * side == tokens
            # the partition in plain Python: a tool may build the model on the meta device (tools/heldout_loss.py), where a
            # tensor has no values
            if route == "pos":   # n tiles of the side x side grid, as square as n allows (n = 8 on 16 x 16: 4 rows x 2 columns of 4 x 8 tiles)
                ry = 2 ** math.ceil(math.log2(n) / 2); rx = n // ry
                assert ry * rx == n and side % ry == 0 and side % rx == 0, (n, side)
                region = [(y // (side // ry)) * rx + x // (side // rx) for y in range(side) for x in range(side)]
            else:
                import random
                perm = list(range(tokens))
                random.Random(20261113).shuffle(perm)
                region = [0] * tokens
                for i, p in enumerate(perm):
                    region[p] = i % n
            order = [[i for i in range(tokens) if region[i] == e] for e in range(n)]   # the token positions of each expert
            self.register_buffer("moe_tokens", torch.tensor(order, dtype=torch.int64))
        self.reset_parameters()

    def reset_parameters(self):
        H, d = self.G * self.h, self.dim   # the active hidden: the bounds of a dense layer of that hidden
        nn.init.uniform_(self.e12, -math.sqrt(6 / (2 * H + d)), math.sqrt(6 / (2 * H + d)))
        nn.init.uniform_(self.e3t, -math.sqrt(6 / (H + d)), math.sqrt(6 / (H + d)))

    # ---- what the optimizer and FSDP2 need to know ----
    def muon_units(self):
        """{parameter name: (row units, True if stored transposed, factor on the MuP coefficient) or None (AdamW)}."""
        out = {"e12": (2 * self.n, False, self.lr_mult), "e3t": (self.n, True, self.lr_mult / self.G)}
        if self.router is not None:
            out["router.weight"] = None
        if self.shared is not None:
            out["shared.w12.weight"] = (2, False, 1.0)
            out["shared.w3.weight"] = (1, False, 1.0)
        return out

    def fsdp_keep(self):
        """Parameters that stay replicated in fp32 under FSDP2 (the router decides in full precision)."""
        return [self.router.weight] if self.router is not None else []

    def extra_repr(self):
        return (f"experts={self.n}, active={self.G}, expert_hidden={self.h}, route={self.route}, capacity={self.capacity}, "
                f"router_in={self.router_in}, shared={self.shared is not None}")

    # ---- the experts on a slot table ----
    def _experts(self, xf, idx, gate):
        """xf [M, d]; idx [n, cap] token of each slot; gate [n, cap] weight of each slot (0 = empty) or None (all 1).
        Returns [M, d]."""
        n, h, d = self.n, self.h, self.dim
        cap = idx.shape[1]
        flat = idx.reshape(-1)
        if _KERNELS == 4 and xf.is_cuda:   # v4: v3's gather; the gate on the hidden side; a plain sum over each token's slots
            from src.models.layers import moe_ops   # noqa: F401  (registers torch.ops.gpic.*)
            valid = (gate.reshape(-1) != 0) if gate is not None else torch.ones_like(flat, dtype=torch.bool)
            inv = torch.ops.gpic.moe_inv(flat, valid, xf.shape[0], n)
            perm, offsets = torch.ops.gpic.moe_csr3(inv, cap)
            xe = torch.ops.gpic.moe_gather3(xf, flat, perm, offsets).view(n, cap, d)
            a1, a2 = _bmm_wt(xe, self.e12.view(n, 2 * h, d)).chunk(2, dim=-1)
            hid = F.silu(a1) * a2
            if gate is not None:   # fp32 gate x bf16 hidden, rounded to the hidden's dtype once (one fused kernel when compiled)
                hid = (hid * gate.unsqueeze(-1)).to(hid.dtype)
            o = torch.bmm(hid, self.e3t.view(n, h, d)).reshape(n * cap, d)
            return torch.ops.gpic.moe_sum4(o, flat, perm, offsets, xf.shape[0])
        if _KERNELS == 3 and xf.is_cuda:   # v3: the same sums over each token's slot list (CSR from the inverse map, no sort)
            from src.models.layers import moe_ops   # noqa: F401  (registers torch.ops.gpic.*)
            valid = (gate.reshape(-1) != 0) if gate is not None else torch.ones_like(flat, dtype=torch.bool)
            inv = torch.ops.gpic.moe_inv(flat, valid, xf.shape[0], n)
            perm, offsets = torch.ops.gpic.moe_csr3(inv, cap)
            xe = torch.ops.gpic.moe_gather3(xf, flat, perm, offsets).view(n, cap, d)
            a1, a2 = _bmm_wt(xe, self.e12.view(n, 2 * h, d)).chunk(2, dim=-1)
            o = torch.bmm(F.silu(a1) * a2, self.e3t.view(n, h, d)).reshape(n * cap, d)
            g = gate.reshape(-1) if gate is not None else o.new_zeros(0, dtype=torch.float32)
            return torch.ops.gpic.moe_combine3(o, g, flat, perm, offsets, xf.shape[0], gate is not None)
        if _KERNELS == 2 and xf.is_cuda:
            from src.models.layers import moe_ops   # noqa: F401  (registers torch.ops.gpic.*)
            valid = (gate.reshape(-1) != 0) if gate is not None else torch.ones_like(flat, dtype=torch.bool)
            inv = torch.ops.gpic.moe_inv(flat, valid, xf.shape[0], n)
            xe = torch.ops.gpic.moe_gather2(xf, flat, inv).view(n, cap, d)
            a1, a2 = _bmm_wt(xe, self.e12.view(n, 2 * h, d)).chunk(2, dim=-1)
            o = torch.bmm(F.silu(a1) * a2, self.e3t.view(n, h, d)).reshape(n * cap, d)
            g = gate.reshape(-1) if gate is not None else o.new_zeros(0, dtype=torch.float32)
            return torch.ops.gpic.moe_combine2(o, g, flat, inv, xf.shape[0], gate is not None)
        if _KERNELS and xf.is_cuda:
            from src.models.layers import moe_ops   # noqa: F401  (registers torch.ops.gpic.*)
            perm, offsets = torch.ops.gpic.moe_csr(flat, xf.shape[0])
            xe = torch.ops.gpic.moe_gather(xf, flat, perm, offsets).view(n, cap, d)
            a1, a2 = _bmm_wt(xe, self.e12.view(n, 2 * h, d)).chunk(2, dim=-1)
            o = torch.bmm(F.silu(a1) * a2, self.e3t.view(n, h, d)).reshape(n * cap, d)
            g = gate.reshape(-1) if gate is not None else o.new_zeros(0, dtype=torch.float32)
            return torch.ops.gpic.moe_combine(o, g, flat, perm, offsets, xf.shape[0], gate is not None)
        # sums over a token's slots in fp32: the combine below, and the gather's backward (the round trip through fp32 makes it
        # a scatter-add into an fp32 gradient). In bf16 a token's G or more expert terms would each round to bf16 (the main
        # line's packed-caption path avoids bf16 atomics for this reason, lever E). The forward values are unchanged:
        # bf16 -> fp32 -> bf16 is exact.
        xe = xf.float().index_select(0, flat).to(xf.dtype).view(n, cap, d)
        a1, a2 = _bmm_wt(xe, self.e12.view(n, 2 * h, d)).chunk(2, dim=-1)
        o = torch.bmm(F.silu(a1) * a2, self.e3t.view(n, h, d))
        o = o.float() if gate is None else o.float() * gate.unsqueeze(-1).float()
        return torch.zeros(xf.shape[0], d, dtype=torch.float32, device=xf.device).index_add_(
            0, flat, o.reshape(n * cap, d)).to(xe.dtype)

    @staticmethod
    def _compact(sel, cap, w=None):
        """sel [M, n] bool -> slot table: idx [n, cap], gate [n, cap], kept [M, n] bool. Each expert keeps its first cap selected
        tokens in token order (a stable sort of the unselected flag along the tokens: no scan over the tokens, which Inductor's
        CUDA codegen cannot fuse with a reduction, 'NYI TritonSplitDimKernel reductions'); a slot past the expert's selected
        tokens holds an unselected token with gate 0. gate: w's value in a kept slot (w = None: 1), else 0. Static shapes."""
        M, n = sel.shape
        st = sel.t()                                                                   # [n, M]
        idx = torch.sort((~st).to(torch.uint8), dim=1, stable=True).indices[:, :cap]   # selected first, each in token order
        taken = st.gather(1, idx)                                                     # [n, cap]: the slot holds a selected token
        gate = taken.to(torch.float32) if w is None else w.t().gather(1, idx) * taken
        kept = torch.zeros(n, M, dtype=torch.bool, device=sel.device).scatter(1, idx, taken).t()   # out of place (Inductor)
        return idx, gate, kept

    def _scores(self, x, xn, cond):
        r = x if self.router_in == "mod" else xn + cond
        if _ROUTER and r.is_cuda:
            from src.models.layers import moe_ops   # noqa: F401  (registers torch.ops.gpic.*)
            return torch.ops.gpic.moe_scores(r.reshape(-1, self.dim), self.router.weight)                # [M, n] fp32
        with torch.autocast(device_type=x.device.type, enabled=False):
            s = F.linear(r.reshape(-1, self.dim).float(), self.router.weight.float())      # [M, n] fp32
        return s

    def _gates(self, sig, sel):
        den = (sig * sel).sum(1, keepdim=True).clamp_min(1e-20)
        return self.G * sig / den

    def _ecb_select(self, sig, capn):
        """Each expert's capn highest-scoring tokens of the micro-batch (torch.topk along the tokens, unsorted: the set is the
        same, and the slot order does not matter): idx [n, capn], and the boundary score per expert (its lowest selected score) [n]."""
        v, i = sig.t().topk(capn, dim=1, sorted=False)
        return i, v.min(dim=1).values

    # ---- forward ----
    def forward(self, x, t=None, cond=None, xn=None):
        """x [B, T, d] (the dense layer's input); t [B] raw timesteps (noise route); cond [B, 1, d] and xn [B, T, d] for
        router_in = plain."""
        if not self.training and self.route in ("tc", "ecb"):
            return self._forward_eval(x, cond, xn)
        B, T, d = x.shape
        idx, gate = self.slots(x, t, cond, xn)
        y = self._experts(x.reshape(B * T, d), idx, gate).view(B, T, d)
        return y if self.shared is None else y + self.shared(x)

    def slots(self, x, t=None, cond=None, xn=None):
        """The slot table of a training forward (also evaluation's for the routes eci, noise, pos, hash): idx [n, cap],
        gate [n, cap] or None. Static shapes."""
        B, T, d = x.shape
        M, n, G = B * T, self.n, self.G
        gate = None
        if self.route == "tc":
            sig = torch.sigmoid(self._scores(x, xn, cond))
            top = (sig + self.moe_bias).topk(G, dim=1).indices
            sel = torch.zeros(M, n, dtype=torch.bool, device=x.device).scatter(1, top, True)
            cap = min(M, -(-int(math.ceil(self.capacity * G * M / n)) // 8) * 8)
            idx, gate, kept = self._compact(sel, cap, self._gates(sig, sel))
            self._stat(sel.sum(0), None, (sel & ~kept).sum(), M)
        elif self.route == "ecb":
            assert (G * M) % n == 0, (G, M, n)
            sig = torch.sigmoid(self._scores(x, xn, cond))
            idx, thr = self._ecb_select(sig, G * M // n)
            sel = torch.zeros(M, n, dtype=torch.bool, device=x.device).scatter(
                0, idx.t(), torch.ones(G * M // n, n, dtype=torch.bool, device=x.device))
            gate = self._gates(sig, sel).t().gather(1, idx)
            self._stat(sel.sum(0), thr, None, M)
        elif self.route == "eci":
            assert (G * T) % n == 0, (G, T, n)
            c = G * T // n
            s = self._scores(x, xn, cond)
            sig = torch.sigmoid(s)
            top = s.view(B, T, n).transpose(1, 2).topk(c, dim=2).indices                          # [B, n, c] tokens in the image
            idx = (top + torch.arange(B, device=x.device)[:, None, None] * T).permute(1, 0, 2).reshape(n, B * c)
            sel = torch.zeros(M, n, dtype=torch.bool, device=x.device).scatter(
                0, idx.t(), torch.ones(B * c, n, dtype=torch.bool, device=x.device))
            gate = self._gates(sig, sel).t().gather(1, idx)
            self._stat(sel.sum(0), None, None, M)
        elif self.route == "noise":
            # [B] expert of each image: the number of edges below t (torch.bucketize's index, right=False; bucketize itself
            # breaks Inductor's CUDA lowering of the load sum below, 'StarDep does not have an index')
            e = (t.reshape(B, 1).float() > self.moe_edges[None, :]).sum(1)
            sel = e[:, None].expand(B, T).reshape(M)[:, None] == torch.arange(n, device=x.device)[None, :]   # [M, n]
            cap = min(M, -(-int(math.ceil(self.capacity * M / n)) // T) * T)
            idx, gate, kept = self._compact(sel, cap)
            self._stat(sel.sum(0), None, (sel & ~kept).sum(), M)
        else:   # pos, hash: the fixed tokens of each expert, in every image
            idx = (self.moe_tokens[:, None, :] + torch.arange(B, device=x.device)[None, :, None] * T).reshape(n, M // n)
        return idx, gate

    def _stat(self, load, thr, drops, tokens):
        if not self.training:
            return
        z = torch.zeros((), dtype=torch.float32, device=load.device)
        v = torch.cat([load.to(torch.float32), thr if thr is not None else torch.zeros_like(load, dtype=torch.float32),
                       torch.stack([z if drops is None else drops.to(torch.float32), z + tokens, z + 1])])
        self.moe_stat.add_(v.detach())

    @torch.compiler.disable
    def _forward_eval(self, x, cond, xn):
        """tc and ecb outside training: every selected (token, expert) pair is computed; the slot table is as wide as the
        busiest expert needs (rounded up to 256 tokens, so the GEMM sizes repeat). Not compiled: the width varies."""
        B, T, d = x.shape
        idx, gate = self.slots_eval(x, cond, xn)
        y = self._experts(x.reshape(B * T, d), idx, gate).view(B, T, d)
        return y if self.shared is None else y + self.shared(x)

    def slots_eval(self, x, cond=None, xn=None):
        B, T, d = x.shape
        M, n, G = B * T, self.n, self.G
        sig = torch.sigmoid(self._scores(x, xn, cond))
        if self.route == "tc":
            top = (sig + self.moe_bias).topk(G, dim=1).indices
            sel = torch.zeros(M, n, dtype=torch.bool, device=x.device).scatter(1, top, True)
        else:
            sel = sig >= self.moe_thr
        cap = max(256, -(-int(sel.sum(0).max()) // 256) * 256)
        idx, gate, _ = self._compact(sel, cap, self._gates(sig, sel))
        return idx, gate
