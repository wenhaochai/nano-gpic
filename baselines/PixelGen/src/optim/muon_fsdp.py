"""MuonAdamW (src/optim/muon.py, the reference) on FSDP2's sharded parameters (MFU lever B, docs/mfu.md "Muon").

Under FSDP2 every parameter, gradient and optimizer state is a DTensor with Shard(0) on one mesh: each rank holds a block
of rows. AdamW and the momentum are elementwise, so they run on the local tensors (as lever B2's ShardLocalAdamW).
Newton-Schulz needs whole matrices: each matrix part ("unit") keeps the reference's owner rank (the same
longest-processing-time assignment, MuonAdamW._build_plan), every rank sends its rows of the Nesterov input of each unit
to the unit's owner (one all_to_all, in the Newton-Schulz dtype: the reference casts to it first thing, so the owner
sees the same input), the owner orthogonalizes its units batched per shape exactly as the reference's owner does, and
one all_to_all returns each rank its rows of every update, which it adds to its rows of the weights with the
reference's scale. With the same owners and the same batches the result equals the reference under DDP bit for bit
(tools/mfu/tests/test_muon_fsdp_cpu.py); the gradients themselves are FSDP2's (reduce-scattered in fp32).
"""
import math

import torch
import torch.distributed as dist
from torch.optim.adam import adam as _adam_fn

from src.optim.muon import MuonAdamW, msign


def _is_dt(t):
    from torch.distributed.tensor import DTensor
    return isinstance(t, DTensor)


def _loc(t):
    return t.to_local() if _is_dt(t) else t


def _a2a(recv, send, recv_sizes, send_sizes):
    """all_to_all_single with sizes in elements; bf16 travels as bytes (gloo has no bf16 or int16 all_to_all): the same
    bits."""
    if send.dtype == torch.bfloat16:
        k = send.element_size()
        recv, send = recv.view(torch.uint8), send.view(torch.uint8)
        recv_sizes, send_sizes = [k * n for n in recv_sizes], [k * n for n in send_sizes]
    dist.all_to_all_single(recv, send, recv_sizes, send_sizes)


class ShardMuonAdamW(MuonAdamW):
    """MuonAdamW for FSDP2 (Shard(0) DTensor parameters on one mesh). Same arguments, state and checkpoint layout."""

    @classmethod
    def like(cls, opt, param_groups):
        g = next((g for g in opt.param_groups if g.get("use_muon")), {})
        d = opt.defaults
        return cls(param_groups, lr=d["lr"], betas=d["betas"], eps=d["eps"], weight_decay=d["weight_decay"],
                   muon_lr=g.get("lr"), muon_weight_decay=g.get("weight_decay"), muon_momentum=g.get("momentum", 0.95),
                   muon_nesterov=g.get("nesterov", True), ns_steps=g.get("ns_steps", 5), split=g.get("split", False),
                   fused_adamw=opt.fused_adamw, ns_dtype=str(opt.ns_dtype).replace("torch.", ""),
                   update_scale=g.get("update_scale", "mup"))

    # ---------------------------------------------------------------------------------------------------------------
    def _rows(self, p, world):
        """Every rank's row block [s0, s1) of p (FSDP2: torch.chunk along dim 0); checks this rank's against its local."""
        R = p.shape[0]
        c = -(-R // world)
        rows = [(min(q * c, R), min((q + 1) * c, R)) for q in range(world)]
        if _is_dt(p):
            assert p.to_local().shape[0] == rows[dist.get_rank()][1] - rows[dist.get_rank()][0], (tuple(p.shape), world)
        return rows

    def _build_shard_plan(self, group):
        plan = self._build_plan(group)   # the reference's units and owners
        world, rank = plan["world"], plan["rank"]
        params = group["params"]
        rows = [self._rows(p, world) for p in params]
        for u in plan["units"]:   # each rank's piece of each unit: the overlap of the unit's rows with the rank's block
            u["pieces"] = [(max(u["r0"], s0), min(u["r1"], s1)) for s0, s1 in rows[u["p"]]]
        units = plan["units"]
        mine = [j for j, u in enumerate(units) if u["owner"] == rank]
        n = lambda u, q: max(0, u["pieces"][q][1] - u["pieces"][q][0]) * u["shape"][1]
        # send: to each destination d, my piece of every unit d owns, in unit order; recv: from each source q, its
        # piece of every unit I own, in unit order (the return trip swaps the two)
        send_sizes = [sum(n(u, rank) for u in units if u["owner"] == d) for d in range(world)]
        recv_sizes = [sum(n(units[j], q) for j in mine) for q in range(world)]
        recv_off = {}   # (unit, source) -> offset in the receive buffer
        off = 0
        for q in range(world):
            for j in mine:
                recv_off[(j, q)] = off
                off += n(units[j], q)
        by_shape = {}
        for j in mine:
            by_shape.setdefault(units[j]["shape"], []).append(j)
        plan.update(rows=rows, send_sizes=send_sizes, recv_sizes=recv_sizes, recv_off=recv_off, mine_ids=mine,
                    by_shape=by_shape, n=n)
        return plan

    @torch.no_grad()
    def _muon_step(self, group):
        params = group["params"]
        if not dist.is_initialized() or dist.get_world_size() == 1 or not any(_is_dt(p) for p in params):
            return super()._muon_step(group)   # unsharded: the reference
        grads, bufs = [], []
        for p in params:
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            st = self.state[p]
            if "momentum_buffer" not in st:
                st["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            grads.append(_loc(g))
            bufs.append(_loc(st["momentum_buffer"]))
        beta, lr, wd = group["momentum"], group["lr"], group["weight_decay"]
        torch._foreach_lerp_(bufs, grads, 1 - beta)                      # m <- beta m + (1 - beta) g, on the shards
        if self._plan is None:
            self._plan = self._build_shard_plan(group)
        plan = self._plan
        world, rank, units, dt = plan["world"], plan["rank"], plan["units"], self.ns_dtype
        dev = bufs[0].device
        # this rank's Nesterov input, per parameter (rows of its block), in the Newton-Schulz dtype
        u_loc = [(g.lerp(m, beta) if group["nesterov"] else m).to(dt) for g, m in zip(grads, bufs)]
        s0 = [plan["rows"][i][rank][0] for i in range(len(params))]
        pieces = []
        for d in range(world):
            for u in units:
                if u["owner"] == d:
                    a, b = u["pieces"][rank]
                    if b > a:
                        pieces.append(u_loc[u["p"]][a - s0[u["p"]]:b - s0[u["p"]]].reshape(-1))
        send = torch.cat(pieces) if pieces else torch.empty(0, dtype=dt, device=dev)
        recv = torch.empty(sum(plan["recv_sizes"]), dtype=dt, device=dev)
        _a2a(recv, send, plan["recv_sizes"], plan["send_sizes"])
        # the owner: whole units from the pieces, Newton-Schulz batched per shape (as the reference's owner)
        n = plan["n"]
        O_of = {}
        for shape, js in plan["by_shape"].items():
            mats = []
            for j in js:
                u = units[j]
                parts = [recv[plan["recv_off"][(j, q)]:plan["recv_off"][(j, q)] + n(u, q)].view(-1, shape[1])
                         for q in range(world) if n(u, q)]
                mats.append(torch.cat(parts, dim=0))
            O = msign(torch.stack(mats), steps=group["ns_steps"], dtype=dt)
            for j, o in zip(js, O):
                O_of[j] = o
        # return trip: to each source q, its rows of every update I made, in the order q expects (unit order)
        back = []
        for q in range(world):
            for j in plan["mine_ids"]:
                u = units[j]
                a, b = u["pieces"][q]
                if b > a:
                    back.append(O_of[j][a - u["r0"]:b - u["r0"]].reshape(-1))
        send2 = torch.cat(back) if back else torch.empty(0, dtype=dt, device=dev)
        recv2 = torch.empty(sum(plan["send_sizes"]), dtype=dt, device=dev)
        _a2a(recv2, send2, plan["send_sizes"], plan["recv_sizes"])
        w_loc = [_loc(p) for p in params]
        if wd:
            torch._foreach_mul_(w_loc, 1 - lr * wd)
        off = 0
        for d in range(world):   # W <- W - lr * coef * O, the reference's add (update_coef), on my rows
            for u in units:
                if u["owner"] == d:
                    a, b = u["pieces"][rank]
                    if b > a:
                        k = (b - a) * u["shape"][1]
                        upd = recv2[off:off + k].view(b - a, u["shape"][1])
                        w_loc[u["p"]][a - s0[u["p"]]:b - s0[u["p"]]].add_(upd, alpha=-lr * self.update_coef(group, u["shape"]))
                        off += k

    @torch.no_grad()
    def _adamw_step(self, group):
        """The reference's AdamW call on the local tensors (DTensor state kept, as lever B2)."""
        params, grads, m1, m2, steps = [], [], [], [], []
        for p in group["params"]:
            if p.grad is None:
                continue
            st = self.state[p]
            if not st:
                st["step"] = (torch.zeros((), dtype=torch.float32, device=_loc(p).device) if self.fused_adamw
                              else torch.tensor(0.0, dtype=torch.float32))
                st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if self.fused_adamw and st["step"].device != _loc(p).device:   # Lightning restores the state on the CPU
                st["step"] = st["step"].to(_loc(p).device, torch.float32)
            params.append(_loc(p)); grads.append(_loc(p.grad)); m1.append(_loc(st["exp_avg"]))
            m2.append(_loc(st["exp_avg_sq"])); steps.append(st["step"])
        if not params:
            return
        b1, b2 = group["betas"]
        _adam_fn(params, grads, m1, m2, [], steps, foreach=None if self.fused_adamw else True, capturable=False,
                 differentiable=False, fused=self.fused_adamw or None, grad_scale=None, found_inf=None,
                 has_complex=False, decoupled_weight_decay=True, amsgrad=False, beta1=b1, beta2=b2, lr=group["lr"],
                 weight_decay=group["weight_decay"], eps=group["eps"], maximize=False)
