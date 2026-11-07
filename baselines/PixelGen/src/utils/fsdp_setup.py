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
# unit. Each original parameter becomes a plain tensor attribute of its module, a view of the flat parameter that a
# forward pre-hook sets after FSDP2's own (which all-gathers the unit): one split per unit per forward (its backward is one
# concatenation into the flat gradient), and the views enter each compiled block as ordinary inputs, so the blocks of one
# class share one graph as before (a first version read the views through per-instance properties: every block got its
# own class, its own graph, and past Dynamo's 8 graphs per function the rest ran eagerly, 3.2x slower at d1024, w1431).
# The same values: the same weights, gathered in bf16 as one tensor, and each gradient element reduced in fp32 as before.
# Checkpoints keep the original keys (FlatModelParallelStrategy splits the flat tensors on save and rebuilds them on load).
FLAT = "_b3_flat"
ALIGN = 16   # every view starts on a 16-element boundary (aligned inputs for the compiled kernels)


@torch.compiler.disable
def _b3_pre_forward(module, args):
    """Set the views of this module's parameters (FSDP2's pre-forward hook has unsharded the unit: prepend=True). The
    unit's first module makes the split; the others reuse it."""
    owner, first, entries = module.__dict__["_b3_spec"]
    if first or "_b3_pieces" not in owner.__dict__:
        flat = owner._parameters[FLAT]
        assert not is_dtensor(flat), "B3: a flat parameter is still sharded in its unit's forward"
        owner.__dict__["_b3_pieces"] = torch.split(flat, owner.__dict__["_b3_sizes"])
    pieces = owner.__dict__["_b3_pieces"]
    for sub, pname, i, shape in entries:
        sub.__dict__[pname] = pieces[i].view(shape)


def flatten_unit(unit, keep, names, trainable_only=True):
    """The (trainable, unless trainable_only=False: the EMA module's are frozen) parameters of the modules in unit, except
    keep, as views of one flat parameter registered on unit[0] (padded so that every view is aligned). names: {module:
    qualified name}. Returns the key map [(parameter key, flat key, offset, numel, shape)]."""
    owner, chunks, sizes, per_module, kmap, off = unit[0], [], [], [], [], 0
    q = lambda m, n: f"{names[m]}.{n}" if names[m] else n
    proto = None
    for m in unit:
        entries = []
        for sub in m.modules():
            for pname, p in list(sub._parameters.items()):
                if p is None or p in keep or (trainable_only and not p.requires_grad):
                    continue
                if off % ALIGN:
                    pad = ALIGN - off % ALIGN
                    chunks.append(p.detach().new_zeros(pad)); sizes.append(pad); off += pad
                entries.append((sub, pname, len(sizes), tuple(p.shape)))
                kmap.append((q(sub, pname), q(owner, FLAT), off, p.numel(), tuple(p.shape)))
                chunks.append(p.detach().reshape(-1)); sizes.append(p.numel()); off += p.numel()
                proto = p
        per_module.append((m, entries))
    if proto is None:
        return []
    owner.register_parameter(FLAT, torch.nn.Parameter(torch.cat(chunks), requires_grad=trainable_only))
    owner.__dict__["_b3_sizes"] = sizes
    for m, entries in per_module:
        for sub, pname, _, _ in entries:
            del sub._parameters[pname]
            sub.__dict__[pname] = proto.new_empty(0)   # a stand-in outside the forward (for `is not None` checks)
        m.__dict__["_b3_spec"] = (owner, m is owner, entries)
        m.register_forward_pre_hook(_b3_pre_forward)   # after FSDP2's (registered with prepend=True)
    return kmap


# ---- MFU lever X (GPIC_MFU_CUDAGRAPH=1): the compiled blocks replayed as CUDA graphs ----
# At d896-d1280 the forward is bound by the host's launches (py-spy, d1280: the main thread is busy for the whole forward
# while the backward, launched from the autograd thread, keeps the GPU busy). Inductor's CUDA graphs (mode
# "reduce-overhead") replay each block's kernels with one launch: the same kernels in the same order, the same values. A
# graph keeps its parameters' addresses, so FSDP2 must keep each unit's unsharded storage between steps (it frees it
# after the backward and allocates it again before the next forward, at whatever address the allocator gives); the
# all-gather still writes the new values into it every step. Costs one bf16 copy of the parameters per GPU (0.75 GB at
# d1024, 1.7 GB at d1792). A new caption length T records a new graph once.
def _keep_unsharded_storage():
    from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
    orig = FSDPParam.free_unsharded_param
    if getattr(orig, "_gpic_keep", False):
        return

    def free_unsharded_param(self):
        from torch.distributed.fsdp._fully_shard._fsdp_param import compiled_autograd_enabled
        if compiled_autograd_enabled():
            return orig(self)
        # keep the storage: the next all-gather copies the new values into the same addresses

    free_unsharded_param._gpic_keep = True
    FSDPParam.free_unsharded_param = free_unsharded_param


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
        graphs = os.environ.get("GPIC_MFU_CUDAGRAPH") == "1"
        if graphs:
            _keep_unsharded_storage()
        for blk in list(net.text_refine_blocks) + list(net.blocks):
            blk.compile(**({"mode": "reduce-overhead"} if graphs else {}))
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
        if k in self.parts:   # zeros in the alignment gaps (flatten_unit's padding)
            parts = sorted(self.parts[k])
            out = self.sd[parts[0][1]].new_zeros(max(o + self.sd[p].numel() for o, p in parts))
            for o, p in parts:
                out[o:o + self.sd[p].numel()] = self.sd[p].reshape(-1)
            return out
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


def _nosync_strategy():
    from contextlib import contextmanager
    from lightning.pytorch.strategies import ModelParallelStrategy

    class NoSyncModelParallelStrategy(ModelParallelStrategy):
        """ModelParallelStrategy with gradient accumulation as DDP has it (new ladders: global batch 1024 = 8 x 32 x 4;
        Lightning blocks the gradient sync only for DDP): in the micro-batches before the last of an optimizer step,
        FSDP2 neither reduce-scatters (it sums the unsharded gradients in its fp32 reduce dtype) nor reshards after the
        backward (the next micro-batch needs no all-gather); the last micro-batch reduce-scatters the sum. The same
        gradient up to fp32 summation order."""

        @contextmanager
        def block_backward_sync(self):
            from torch.distributed.fsdp import FSDPModule
            mods = [m for m in self.model.modules() if isinstance(m, FSDPModule)]
            for m in mods:
                m.set_requires_gradient_sync(False, recurse=False)
                m.set_reshard_after_backward(False, recurse=False)
            try:
                yield None
            finally:
                for m in mods:
                    m.set_requires_gradient_sync(True, recurse=False)
                    m.set_reshard_after_backward(True, recurse=False)

    return NoSyncModelParallelStrategy


_FLAT_STRATEGY = []
_NOSYNC_STRATEGY = []


def __getattr__(name):   # `src.utils.fsdp_setup.<Strategy>` for the trainer's strategy JSON, built once
    if name == "FlatModelParallelStrategy":
        if not _FLAT_STRATEGY:
            _FLAT_STRATEGY.append(_flat_strategy())
        return _FLAT_STRATEGY[0]
    if name == "NoSyncModelParallelStrategy":
        if not _NOSYNC_STRATEGY:
            _NOSYNC_STRATEGY.append(_nosync_strategy())
        return _NOSYNC_STRATEGY[0]
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
    # one multi-tensor copy instead of one kernel per gradient (about 68 at d1280: py-spy 1.5% of the main thread)
    torch._foreach_copy_(grads, torch._utils._unflatten_dense_tensors(flat, grads))


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
