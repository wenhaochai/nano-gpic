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
         scores (found with a 4096-bin histogram, no sort; ties inside the boundary bin go by token order). A token gets
         0 to n experts. The boundary score per expert is averaged (EMA, in the callback) into moe_thr; in evaluation a
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

import torch
import torch.nn as nn
import torch.nn.functional as F

BINS = 4096
ROUTES = ("tc", "ecb", "eci", "noise", "pos", "hash")


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
            if route == "pos":   # n tiles of the side x side grid, as square as n allows (n = 8 on 16 x 16: 4 rows x 2 columns of 4 x 8 tiles)
                ry = 2 ** math.ceil(math.log2(n) / 2); rx = n // ry
                assert ry * rx == n and side % ry == 0 and side % rx == 0, (n, side)
                yy, xx = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
                region = (yy // (side // ry)) * rx + xx // (side // rx)
                region = region.reshape(-1)
            else:
                g = torch.Generator().manual_seed(20261113)
                region = torch.empty(tokens, dtype=torch.long)
                region[torch.randperm(tokens, generator=g)] = torch.arange(tokens) % n
            order = torch.argsort(region, stable=True).view(n, tokens // n)   # the token positions of each expert
            self.register_buffer("moe_tokens", order)
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
        xe = xf.index_select(0, flat).view(n, cap, d)
        a1, a2 = torch.bmm(xe, self.e12.view(n, 2 * h, d).mT).chunk(2, dim=-1)
        o = torch.bmm(F.silu(a1) * a2, self.e3t.view(n, h, d))
        if gate is not None:
            o = o * gate.unsqueeze(-1).to(o.dtype)
        return o.new_zeros(xf.shape[0], d).index_add_(0, flat, o.reshape(n * cap, d))   # o's dtype (bf16 under autocast)

    @staticmethod
    def _compact(sel, cap, w=None):
        """sel [M, n] bool -> slot table: idx [n, cap] (0 in an empty slot), gate [n, cap] (w's value in a filled slot, else 0;
        w = None: 1), kept [M, n] bool. Each expert keeps its first cap selected tokens, in token order. Static shapes."""
        M, n = sel.shape
        pos = sel.cumsum(0, dtype=torch.int32) - 1
        kept = sel & (pos < cap)
        base = torch.arange(n, device=sel.device, dtype=torch.int64) * cap
        slot = torch.where(kept, pos.to(torch.int64) + base, n * cap).reshape(-1)        # n * cap: a scratch slot
        tok = torch.arange(M, device=sel.device, dtype=torch.int64)[:, None].expand(M, n).reshape(-1)
        idx = torch.zeros(n * cap + 1, dtype=torch.int64, device=sel.device).scatter_(0, slot, tok)[:n * cap].view(n, cap)
        src = kept.to(torch.float32) if w is None else w * kept
        gate = torch.zeros(n * cap + 1, dtype=torch.float32, device=sel.device).scatter(0, slot, src.reshape(-1))
        # the scratch slot may hold any of the unkept values: zero it by construction (unkept sources are 0)
        return idx, gate[:n * cap].view(n, cap), kept

    def _scores(self, x, xn, cond):
        r = x if self.router_in == "mod" else xn + cond
        with torch.autocast(device_type=x.device.type, enabled=False):
            s = F.linear(r.reshape(-1, self.dim).float(), self.router.weight.float())      # [M, n] fp32
        return s

    def _gates(self, sig, sel):
        den = (sig * sel).sum(1, keepdim=True).clamp_min(1e-20)
        return self.G * sig / den

    def _ecb_select(self, sig, capn):
        """Each expert's capn highest tokens of the micro-batch, without a sort: sel_hi = above the boundary bin, sel_bd =
        the boundary bin (filled in token order). Returns sel [M, n] with exactly capn per expert, thr [n] (boundary score)."""
        M, n = sig.shape
        b = (sig * BINS).to(torch.int64).clamp_(0, BINS - 1)
        base = torch.arange(n, device=sig.device, dtype=torch.int64) * BINS
        hist = torch.zeros(n * BINS, dtype=torch.float32, device=sig.device).scatter_add_(
            0, (b + base).reshape(-1), torch.ones(M * n, dtype=torch.float32, device=sig.device)).view(n, BINS)
        above = hist.flip(1).cumsum(1).flip(1)                      # above[e, j] = tokens of expert e in bins >= j
        jb = (above >= capn).sum(1) - 1                             # the boundary bin: the highest j with above >= capn
        hi = b > jb
        n_hi = hi.sum(0)
        bd = b == jb
        take = bd & ((bd.cumsum(0, dtype=torch.int32) - 1) < (capn - n_hi))
        return hi | take, jb.to(torch.float32) / BINS

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
            sel = torch.zeros(M, n, dtype=torch.bool, device=x.device).scatter_(1, top, True)
            cap = min(M, -(-int(math.ceil(self.capacity * G * M / n)) // 8) * 8)
            idx, gate, kept = self._compact(sel, cap, self._gates(sig, sel))
            self._stat(sel.sum(0), None, (sel & ~kept).sum(), M)
        elif self.route == "ecb":
            assert (G * M) % n == 0, (G, M, n)
            sig = torch.sigmoid(self._scores(x, xn, cond))
            sel, thr = self._ecb_select(sig, G * M // n)
            idx, gate, _ = self._compact(sel, G * M // n, self._gates(sig, sel))
            self._stat(sel.sum(0), thr, None, M)
        elif self.route == "eci":
            assert (G * T) % n == 0, (G, T, n)
            c = G * T // n
            s = self._scores(x, xn, cond)
            sig = torch.sigmoid(s)
            top = s.view(B, T, n).transpose(1, 2).topk(c, dim=2).indices                          # [B, n, c] tokens in the image
            idx = (top + torch.arange(B, device=x.device)[:, None, None] * T).permute(1, 0, 2).reshape(n, B * c)
            sel = torch.zeros(M, n, dtype=torch.bool, device=x.device).scatter_(
                0, idx.t(), torch.ones(B * c, n, dtype=torch.bool, device=x.device))
            gate = self._gates(sig, sel).t().gather(1, idx)
            self._stat(sel.sum(0), None, None, M)
        elif self.route == "noise":
            e = torch.bucketize(t.reshape(B).float(), self.moe_edges)                             # [B] expert of each image
            sel = (e[:, None] == torch.arange(n, device=x.device)[None, :])[:, None, :].expand(B, T, n).reshape(M, n)
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
            sel = torch.zeros(M, n, dtype=torch.bool, device=x.device).scatter_(1, top, True)
        else:
            sel = sig >= self.moe_thr
        cap = max(256, -(-int(sel.sum(0).max()) // 256) * 256)
        idx, gate, _ = self._compact(sel, cap, self._gates(sig, sel))
        return idx, gate
