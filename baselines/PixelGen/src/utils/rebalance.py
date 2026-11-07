"""MFU lever E (docs/mfu.md): packed captions, and the samples of each step dealt over the ranks by caption length.

Why: each rank pads its batch to its own longest caption, and DDP's step waits for the slowest rank. A batch of long
captions (T 120-320, 13% of batches) costs its rank about 25-40% more; with 8 ranks about two thirds of the steps hold
one, so the step costs 26% more than with one T for all ranks at d640, 7% at d1280 (w1316).
What: the collate packs a batch's caption tokens back to back (no padding rows); the host plan deals, at every batch
position i, the 8 samples there (one per rank) to the 8 ranks, longest first to the rank with the fewest caption tokens so
far; one all_to_all moves the images and one the caption tokens. A sample keeps its batch position, so it keeps its t and
noise (the official seeding draws them alike on every rank), and every rank keeps 32 samples, so the loss and the
gradient are the same sums over the same 256 samples (up to fp32 summation order).
The host knows every rank's lengths through one gloo all_gather per step (lengths and caption types, a few hundred
bytes); the index tensors are built on the device from a pinned int32 buffer, so no step waits on a pageable copy.
"""
import torch
import torch.distributed as dist


def plan(all_lens, me, world, B):
    """The deal (see module doc). all_lens[r][i]: caption length of sample i of rank r. Returns this rank's send order
    (positions, grouped by destination), the per-destination sample counts, the received (source, position) list in
    all_to_all order, and the received lengths in position order."""
    load = [0] * world
    dst = {}
    for i in sorted(range(B), key=lambda i: -max(all_lens[r][i] for r in range(world))):
        col = sorted(range(world), key=lambda r: -all_lens[r][i])
        order = sorted(range(world), key=lambda r: load[r])
        for r, d in zip(col, order):
            dst[(r, i)] = d
            load[d] += all_lens[r][i]
    send = sorted(range(B), key=lambda i: (dst[(me, i)], i))
    in_s = [sum(dst[(me, i)] == d for i in range(B)) for d in range(world)]
    recv = [(r, i) for r in range(world) for i in range(B) if dst[(r, i)] == me]
    out_s = [sum(1 for (r, _) in recv if r == src) for src in range(world)]
    assert sorted(i for _, i in recv) == list(range(B)), "one sample per position"
    return send, in_s, out_s, recv


BUCKET = 128


def bucket(S):
    """The packed caption rows: the real token count rounded up to a multiple of BUCKET (JiT_T2I.py, y_pack)."""
    return max(BUCKET, -(-S // BUCKET) * BUCKET)


def _excl_cumsum(x):
    return torch.cumsum(x, 0) - x


class Rebalancer:
    """Host side (before the batch moves to the device): host_plan(lens, caption_types). Device side (in the training
    step): exchange(x, y_packed, p) -> images, packed caption tokens and y_pack (src/models/transformer/JiT_T2I.py)."""

    def __init__(self, rebalance=True, pad_multiple=8, max_T=320, ring=4):
        self.rebalance, self.pad_multiple, self.max_T = rebalance, pad_multiple, max_T
        self.gloo = None
        self.ring, self.k = ring, 0
        self.bufs = None

    def _pinned(self, B):
        if self.bufs is None:
            pin = torch.cuda.is_available()   # pinned: the copy to the device is asynchronous (CPU tests: plain memory)
            self.bufs = [torch.empty(5, B, dtype=torch.int32, pin_memory=pin) for _ in range(self.ring)]
        buf = self.bufs[self.k % self.ring]   # a ring: the copy of step s may still run when step s+1 is planned
        self.k += 1
        return buf

    def host_plan(self, lens, caption_types=None):
        B = len(lens)
        on = self.rebalance and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        if not on:
            new_lens, send, inv, starts = list(lens), list(range(B)), list(range(B)), None
            p = {"S_old": sum(lens), "S_new": sum(lens), "types": caption_types}
        else:
            world, me = dist.get_world_size(), dist.get_rank()
            if self.gloo is None:
                self.gloo = dist.new_group(backend="gloo")
            objs = [None] * world
            dist.all_gather_object(objs, (list(lens), caption_types), group=self.gloo)
            all_lens = [o[0] for o in objs]
            send, in_s, out_s, recv = plan(all_lens, me, world, B)
            pos_of = [i for (_, i) in recv]
            inv = [pos_of.index(q) for q in range(B)]
            new_lens = [all_lens[r][i] for (r, i) in sorted(recv, key=lambda ri: ri[1])]
            tin, k = [], 0
            for d in range(world):
                tin.append(sum(lens[q] for q in send[k:k + in_s[d]]))
                k += in_s[d]
            tout = [sum(all_lens[r][q] for (rr, q) in recv if rr == r) for r in range(world)]
            st, off = {}, 0
            for (r, q) in recv:   # received tokens come by (source rank, position): each sample's first row
                st[q] = off
                off += all_lens[r][q]
            starts = [st[q] for q in range(B)]
            p = {"S_old": sum(lens), "S_new": sum(new_lens), "in_s": in_s, "out_s": out_s, "tin": tin, "tout": tout,
                 "types": None if caption_types is None else [objs[r][1][q] for (r, q) in sorted(recv, key=lambda ri: ri[1])]}
        buf = self._pinned(B)
        buf[0] = torch.tensor(send, dtype=torch.int32)
        buf[1] = torch.tensor(inv, dtype=torch.int32)
        buf[2] = torch.tensor(list(lens), dtype=torch.int32)
        buf[3] = torch.tensor(new_lens, dtype=torch.int32)
        buf[4] = torch.tensor(starts if starts is not None else [0] * B, dtype=torch.int32)
        p["idx"] = buf
        p["Tp"] = min(-(-max(new_lens) // self.pad_multiple) * self.pad_multiple, self.max_T)
        p["exchange"] = on
        return p

    def exchange(self, x, y, p):
        dev = x.device
        idx = p["idx"].to(dev, non_blocking=True) if not p["idx"].is_cuda else p["idx"]
        send, inv, old_lens, new_lens, starts = idx[0].long(), idx[1].long(), idx[2].long(), idx[3].long(), idx[4].long()
        B, S_old, S_new = x.shape[0], p["S_old"], p["S_new"]
        if p["exchange"]:
            xr = torch.empty_like(x)
            dist.all_to_all_single(xr.view(B, -1), x.index_select(0, send).view(B, -1), p["out_s"], p["in_s"])
            x = xr.index_select(0, inv)
            ls = old_lens.index_select(0, send)
            # the rows of each sent sample, in send order: its first row in y plus 0..len-1
            first = _excl_cumsum(old_lens).index_select(0, send)
            ar = torch.arange(S_old, device=dev)
            tsend = torch.repeat_interleave(first - _excl_cumsum(ls), ls, output_size=S_old) + ar
            yr = y.new_empty(S_new, y.shape[1])
            dist.all_to_all_single(yr, y.index_select(0, tsend), p["tout"], p["tin"])
            ar = torch.arange(S_new, device=dev)
            tperm = torch.repeat_interleave(starts - _excl_cumsum(new_lens), new_lens, output_size=S_new) + ar
            y = yr.index_select(0, tperm)
        Tp = p["Tp"]
        S_pad = bucket(S_new)
        cu = torch.zeros(B + 1, dtype=torch.int32, device=dev)
        cu[1:] = torch.cumsum(new_lens, 0).int()
        tok_b = torch.repeat_interleave(torch.arange(B, device=dev), new_lens, output_size=S_new)
        pos = torch.arange(S_new, device=dev) - cu[:-1].long().index_select(0, tok_b)
        nd = S_pad - S_new   # dummy rows: no sample (tok_b = B), position 0, the scratch row of the padded layout
        tok_b = torch.cat([tok_b, tok_b.new_full((nd,), B)])
        pos = torch.cat([pos, pos.new_zeros(nd)])
        pidx = torch.where(tok_b < B, tok_b * Tp + pos, B * Tp)
        y = torch.cat([y, y.new_zeros(nd, y.shape[1])])
        kmask = torch.arange(Tp, device=dev)[None] < new_lens[:, None]   # its shape carries Tp (a dynamic size, not an int)
        onehot = tok_b[:, None] == torch.arange(B, device=dev)[None]
        pack = {"lens": new_lens.int(), "cu": cu, "tok_b": tok_b, "pos": pos, "pidx": pidx, "kmask": kmask,
                "onehot": onehot, "real": tok_b < B, "max_T": self.max_T}
        return x, y, pack
