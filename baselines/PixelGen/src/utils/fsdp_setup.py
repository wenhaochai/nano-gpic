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


def shard(net, mesh, compute_bf16, compile_blocks, group=None):
    """fully_shard the text and main blocks, then the root. Returns the replicated (ignored) parameters.
    group (GPIC_MFU_FSDP_GROUP, default 1): consecutive blocks per FSDP2 unit (a list of modules is one unit: one
    all-gather, one reduce-scatter, one set of hooks), text and main blocks apart. Fewer units, less host work per step;
    the same math (the same parameters gathered, the same gradients reduced)."""
    import os
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16 if compute_bf16 else None, reduce_dtype=torch.float32,
                              cast_forward_inputs=False)   # the blocks' RoPE tables and masks stay as they are
    keep = _keep(net)
    g = int(group or os.environ.get("GPIC_MFU_FSDP_GROUP", "1"))
    for blocks in (list(net.text_refine_blocks), list(net.blocks)):
        for i in range(0, len(blocks), g):
            unit = blocks[i:i + g]
            own = keep & {p for b in unit for p in b.parameters()}
            fully_shard(unit if len(unit) > 1 else unit[0], mesh=mesh, mp_policy=mp, reshard_after_forward=False,
                        ignored_params=own)
    if compile_blocks:
        for blk in list(net.text_refine_blocks) + list(net.blocks):
            blk.compile()
    fully_shard(net, mesh=mesh, mp_policy=mp, reshard_after_forward=False, ignored_params=keep)
    return [p for p in net.parameters() if p in keep]


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
def clip_grad_norm_(params, max_norm):
    """torch.nn.utils.clip_grad_norm_ over sharded (DTensor) and replicated parameters: the 2-norm of all gradients,
    each element counted once (shards all-reduced, replicated parameters counted on every rank once), then the same
    clip_coef = max_norm / (norm + 1e-6), clamped at 1."""
    sh = [p.grad.to_local() for p in params if p.grad is not None and is_dtensor(p.grad)]
    rp = [p.grad for p in params if p.grad is not None and not is_dtensor(p.grad)]
    dev = (sh or rp)[0].device
    sq = torch.stack(torch._foreach_norm(sh)).pow(2).sum() if sh else torch.zeros((), device=dev)
    dist.all_reduce(sq)
    if rp:
        sq = sq + torch.stack(torch._foreach_norm(rp)).pow(2).sum()
    norm = sq.sqrt()
    coef = (max_norm / (norm + 1e-6)).clamp(max=1.0)
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
