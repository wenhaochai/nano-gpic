"""MFU lever B (docs/mfu.md): FSDP2 in place of DDP, with the same math (GPIC_MFU_FSDP=1, ModelParallelStrategy).

- Each block of the denoiser is its own FSDP2 unit (fully_shard), so its parameters are all-gathered just before its
  forward and its gradients reduce-scattered just after its backward, overlapping the neighbouring blocks' compute.
- Compute parameters in bf16 (MixedPrecisionPolicy param_dtype): autocast casts every Linear weight and bias to bf16
  anyway, and a bf16 copy of an fp32 value is the same round-to-nearest either way, so the forward and the bf16 weight
  gradients are unchanged; gradients reduce in fp32 (reduce_dtype), as DDP's fp32 buckets do. The cast moves from the
  forward of every step to the all-gather input (1/8 of the parameters), and nothing casts weights in the forward.
- Kept out of FSDP (ignored_params, fp32, replicated, gradients all-reduced here): the RMSNorm weights, which autocast
  uses in fp32 (a bf16 copy would change them), and pos_embed (a fixed table added in fp32).
- AdamW (fused) and the EMA run on each rank's 1/8 shard: their memory traffic, 11 ms of a d1280 step and 28 + 7 ms of a
  d1792 step under DDP, is divided by 8. The clip is the same global 2-norm over all parameters.
- The EMA module is sharded the same way (fp32, never run in training), so a checkpoint gathers it in full with the
  denoiser (ModelParallelStrategy, save_distributed_checkpoint=False): ms and last checkpoints keep their keys.
"""
import torch
import torch.distributed as dist
from torch.optim.optimizer import _use_grad_for_differentiable


def _keep(net):
    from src.models.layers.rmsnorm import RMSNorm
    keep = {p for m in net.modules() if isinstance(m, RMSNorm) for p in m.parameters()}
    if hasattr(net, "pos_embed"):
        keep.add(net.pos_embed)
    return keep


# ---- MFU lever B3 (GPIC_MFU_FSDP_FLAT=1): one flat parameter per FSDP2 unit ----
# FSDP2 does Python work per parameter on every step (the all-gather's copy-in and copy-out lists, a DTensor for every
# sharded gradient, the swap of every parameter between its sharded and unsharded tensor: about a third of the host's
# time at d1024-d1280, where the host binds; docs/mfu.md B3). With one flat fp32 parameter per unit, that work is per
# unit. Each original parameter becomes a view of the flat one, read through a property of its module (torch.compile
# traces the slice), so the forward and backward compute the same values: the same weights, gathered in bf16 as one
# tensor, and each gradient element reduced in fp32 as before. Checkpoints keep the original keys
# (FlatModelParallelStrategy splits the flat tensors on save and rebuilds them on load).
FLAT = "_b3_flat"


def _b3_view(self, name):
    o, n, shape = self.__dict__["_b3_views"][name]
    flat = self.__dict__["_b3_owner"]._parameters[FLAT]
    if is_dtensor(flat):   # sharded (outside the unit's forward and backward): no full weight exists here; a stand-in
        return torch.empty(0, dtype=flat.dtype)   # for `is not None` checks (a slice of a DTensor would redistribute)
    return flat[o:o + n].view(shape)


def _install_view(sub, name, owner, o, n, shape):
    if "_b3_views" not in sub.__dict__:   # this instance gets its own subclass, so the property is its alone
        cls = sub.__class__
        sub.__class__ = type(cls.__name__, (cls,), {"__module__": cls.__module__})
        sub.__dict__["_b3_views"] = {}
        sub.__dict__["_b3_owner"] = owner   # not registered as a submodule
    assert sub.__dict__["_b3_owner"] is owner
    sub.__dict__["_b3_views"][name] = (o, n, tuple(shape))
    setattr(type(sub), name, property(lambda self, _n=name: _b3_view(self, _n)))


def flatten_unit(unit, keep, names, trainable_only=True):
    """The (trainable, unless trainable_only=False: the EMA module's are frozen) parameters of the modules in unit, except
    keep, as views of one flat parameter registered on unit[0]. names: {module: qualified name}. Returns the key map
    [(parameter key, flat key, offset, numel, shape)]."""
    owner, entries, chunks, off = unit[0], [], [], 0
    for m in unit:
        for sub in m.modules():
            for pname, p in list(sub._parameters.items()):
                if p is None or p in keep or (trainable_only and not p.requires_grad):
                    continue
                entries.append((sub, pname, tuple(p.shape), off, p.numel()))
                chunks.append(p.detach().reshape(-1))
                off += p.numel()
    if not entries:
        return []
    owner.register_parameter(FLAT, torch.nn.Parameter(torch.cat(chunks), requires_grad=entries and trainable_only))
    q = lambda m, n: f"{names[m]}.{n}" if names[m] else n
    kmap = []
    for sub, pname, shape, o, n in entries:
        del sub._parameters[pname]
        _install_view(sub, pname, owner, o, n, shape)
        kmap.append((q(sub, pname), q(owner, FLAT), o, n, shape))
    return kmap


def shard(net, mesh, compute_bf16, compile_blocks, group=None, names=None, trainable_only=True):
    """fully_shard the text and main blocks, then the root. Returns the replicated (ignored) parameters.
    group (GPIC_MFU_FSDP_GROUP, default 1): consecutive blocks per FSDP2 unit (a list of modules is one unit: one
    all-gather, one reduce-scatter, one set of hooks), text and main blocks apart. Fewer units, less host work per step;
    the same math (the same parameters gathered, the same gradients reduced).
    With GPIC_MFU_FSDP_FLAT=1 (lever B3) each unit's parameters become one flat parameter first; the key map goes to
    net._b3_kmap (keys qualified by names, {module: name} of the module that is checkpointed; default net's own)."""
    import os
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16 if compute_bf16 else None, reduce_dtype=torch.float32,
                              cast_forward_inputs=False)   # the blocks' RoPE tables and masks stay as they are
    keep = _keep(net)
    g = int(group or os.environ.get("GPIC_MFU_FSDP_GROUP", "1"))
    flat = os.environ.get("GPIC_MFU_FSDP_FLAT") == "1"
    names = names or {m: n for n, m in net.named_modules()}
    kmap = []
    for blocks in (list(net.text_refine_blocks), list(net.blocks)):
        for i in range(0, len(blocks), g):
            unit = blocks[i:i + g]
            if flat:
                kmap += flatten_unit(unit, keep, names, trainable_only)
            own = keep & {p for b in unit for p in b.parameters()}
            fully_shard(unit if len(unit) > 1 else unit[0], mesh=mesh, mp_policy=mp, reshard_after_forward=False,
                        ignored_params=own)
    if compile_blocks:
        for blk in list(net.text_refine_blocks) + list(net.blocks):
            blk.compile()
    fully_shard(net, mesh=mesh, mp_policy=mp, reshard_after_forward=False, ignored_params=keep)
    net.__dict__["_b3_kmap"] = kmap
    return [p for p in net.parameters() if p in keep]


def split_flat(sd, kmap):
    """A full state dict with flat keys -> the same tensors under the original parameter keys (own storages)."""
    out = dict(sd)
    for pkey, fkey, o, n, shape in kmap:
        if fkey in sd:
            out[pkey] = sd[fkey][o:o + n].view(shape).clone()
    for fkey in {k[1] for k in kmap}:
        out.pop(fkey, None)
    return out


class _FlatView:
    """A checkpoint's state_dict seen with flat keys: a flat parameter's key yields the concatenation of its original
    parameters (built when asked for), every other key the stored tensor."""

    def __init__(self, sd, kmap):
        self.sd, self.parts = sd, {}
        for pkey, fkey, o, n, shape in kmap:
            self.parts.setdefault(fkey, []).append((o, pkey))

    def __contains__(self, k):
        return k in self.parts or k in self.sd

    def __getitem__(self, k):
        if k in self.parts:
            return torch.cat([self.sd[p].reshape(-1) for _, p in sorted(self.parts[k])])
        return self.sd[k]


def _flat_strategy():
    from lightning.pytorch.strategies import ModelParallelStrategy

    class FlatModelParallelStrategy(ModelParallelStrategy):
        """ModelParallelStrategy for lever B3: checkpoints keep the original per-parameter keys (the eval path and runs
        without B3 read them); the flat parameters are split on save and rebuilt on load."""

        def _kmap(self):
            m = self.lightning_module
            return getattr(m, "_b3_kmap", None) if m is not None else None

        def lightning_module_state_dict(self):
            sd = super().lightning_module_state_dict()
            kmap = self._kmap()
            if not kmap or not sd:   # with full_state_dict only rank 0 gets the tensors
                return sd
            return split_flat(sd, kmap)

        def load_checkpoint(self, checkpoint_path, weights_only=None):
            kmap = self._kmap()
            if not kmap:
                return super().load_checkpoint(checkpoint_path, weights_only)
            real = torch.load

            def load(*a, **k):
                ck = real(*a, **k)
                if isinstance(ck, dict) and isinstance(ck.get("state_dict"), dict):
                    ck["state_dict"] = _FlatView(ck["state_dict"], kmap)
                return ck
            torch.load = load   # Lightning's _load_checkpoint reads the file with torch.load
            try:
                return super().load_checkpoint(checkpoint_path, weights_only)
            finally:
                torch.load = real

    return FlatModelParallelStrategy


_FLAT_STRATEGY = []


def __getattr__(name):   # `src.utils.fsdp_setup.FlatModelParallelStrategy` for the trainer's strategy JSON, built once
    if name == "FlatModelParallelStrategy":
        if not _FLAT_STRATEGY:
            _FLAT_STRATEGY.append(_flat_strategy())
        return _FLAT_STRATEGY[0]
    raise AttributeError(name)


def is_dtensor(p):
    from torch.distributed.tensor import DTensor
    return isinstance(p, DTensor)


@torch.no_grad()
def sync_replicated_grads(params):
    """All-reduce (mean) the gradients of the replicated parameters, in one flat buffer."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.AVG)
    for g, f in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
        g.copy_(f)


@torch.no_grad()
def clip_grad_norm_(params, max_norm, apply=True):
    """torch.nn.utils.clip_grad_norm_ over sharded (DTensor) and replicated parameters: the 2-norm of all gradients,
    each element counted once (shards all-reduced, replicated parameters counted on every rank once), then the same
    clip_coef = max_norm / (norm + 1e-6), clamped at 1. apply=False (lever H): returns (norm, clip_coef) without the
    multiply, which the fused optimizer kernel does."""
    sh = [p.grad.to_local() for p in params if p.grad is not None and is_dtensor(p.grad)]
    rp = [p.grad for p in params if p.grad is not None and not is_dtensor(p.grad)]
    dev = (sh or rp)[0].device
    sq = torch.stack(torch._foreach_norm(sh)).pow(2).sum() if sh else torch.zeros((), device=dev)
    dist.all_reduce(sq)
    if rp:
        sq = sq + torch.stack(torch._foreach_norm(rp)).pow(2).sum()
    norm = sq.sqrt()
    coef = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    if not apply:
        return norm, coef
    if sh:
        torch._foreach_mul_(sh, coef)
    if rp:
        torch._foreach_mul_(rp, coef)
    return norm


def _local(ts):
    from torch.distributed.tensor import DTensor
    return [t._local_tensor if isinstance(t, DTensor) else t for t in ts]


class ShardLocalAdamW(torch.optim.AdamW):
    """MFU lever B2 (GPIC_MFU_LOCAL_OPT=1, docs/mfu.md): AdamW's step on the local shards of the DTensor parameters. Adam.step
    of torch 2.7 line for line (the same _init_group, so the state holds the same DTensors and the checkpoints keep their
    layout), except that torch.optim.adam.adam receives each DTensor's local tensor: the same fused kernel on the same data,
    without DTensor's dispatch of the 4 x 129-tensor op (CPU: 14.8 against 4.9 ms per step at d1024's sizes). Every
    placement here is Shard(0) on one mesh, where DTensor runs exactly this local op."""

    @classmethod
    def like(cls, opt, param_groups):
        d = opt.defaults
        return cls(param_groups, lr=d["lr"], betas=d["betas"], eps=d["eps"], weight_decay=d["weight_decay"],
                   amsgrad=d["amsgrad"], maximize=d["maximize"], foreach=d["foreach"], capturable=d["capturable"],
                   differentiable=d["differentiable"], fused=d["fused"])

    @_use_grad_for_differentiable
    def step(self, closure=None):
        from torch.optim.adam import adam
        self._cuda_graph_capture_health_check()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps = [], [], [], [], [], []
            beta1, beta2 = group["betas"]
            has_complex = self._init_group(group, params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                                           state_steps)
            adam(_local(params_with_grad), _local(grads), _local(exp_avgs), _local(exp_avg_sqs), _local(max_exp_avg_sqs),
                 state_steps, amsgrad=group["amsgrad"], has_complex=has_complex, beta1=beta1, beta2=beta2, lr=group["lr"],
                 weight_decay=group["weight_decay"], eps=group["eps"], maximize=group["maximize"], foreach=group["foreach"],
                 capturable=group["capturable"], differentiable=group["differentiable"], fused=group["fused"],
                 grad_scale=getattr(self, "grad_scale", None), found_inf=getattr(self, "found_inf", None),
                 decoupled_weight_decay=group["decoupled_weight_decay"])
        return loss
