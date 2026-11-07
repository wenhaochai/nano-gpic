"""MFU lever H (GPIC_MFU_FUSED_OPT, docs/mfu.md): the gradient clip's multiply, AdamW and (=2) the EMA lerp in one Triton
kernel over every parameter, in place of torch's _foreach_mul_ (clip), _fused_adamw_ and _foreach_lerp_ (SimpleEMA).

The arithmetic is torch 2.7's, operation for operation (ATen/native/cuda/fused_adam_utils.cuh adam_math, Lerp.h lerp):
- g = grad * clip_coef in fp32 (clip_grad_norm_'s _foreach_mul_, whose result Adam then reads from memory).
- AdamW in torch's mixed precision: lr, beta1, beta2, weight_decay, eps are doubles, so exp_avg = beta1 * exp_avg +
  (1 - beta1) * g and exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * g * g are computed in double and rounded to fp32;
  bias_correction1 = 1 - beta1^step and sqrt(1 - beta2^step) are computed in double and rounded to fp32; step_size =
  fp32(lr / bias_correction1); denom = fp32(fp32(sqrtf(exp_avg_sq) / bias_correction2_sqrt) + eps); param -= step_size *
  exp_avg / denom in fp32 with IEEE division and square root (PyTorch is not built with fast math). nvcc contracts a * b + c
  into an fma by default; FMA_* below select where this kernel does (the test tools/mfu/tests/test_fused_adamw_ema.py picks
  the variant that equals torch bit for bit).
- EMA (=2): ema = ema + w * (p_new - ema), w = fp32(1 - decay) (lerp with a small weight), on the updated parameter, as
  SimpleEMA's lerp after the optimizer step reads it.
What it saves: the clip's extra pass over the gradients (read and write), the EMA's second read of the parameters, and
torch's multi-tensor launches (320 blocks of 512 threads each, about 1.3 TB/s at these sizes, w1382) for one kernel with a
block per 4096 elements. The optimizer state stays AdamW's (exp_avg, exp_avg_sq, step per parameter), so checkpoints and
resumes are unchanged. Works on plain tensors (DDP) and on the local shards of DTensors (FSDP2, as ShardLocalAdamW).
"""
import torch
import triton
import triton.language as tl
from torch.optim.optimizer import _use_grad_for_differentiable

try:
    from triton.language.extra import libdevice
except ImportError:   # older Triton
    from triton.language.extra.cuda import libdevice

BLOCK = 4096


@triton.jit
def _adamw_ema_kernel(PP, GP, MP, VP, SP, EP, NUMEL, TID, TOFF, COEF, HP, ema_w,
                      HAS_COEF: tl.constexpr, HAS_EMA: tl.constexpr, HAS_WD: tl.constexpr,
                      FMA_M: tl.constexpr, FMA_V: tl.constexpr, FMA_E: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    t = tl.load(TID + pid)
    off = tl.load(TOFF + pid)
    n = tl.load(NUMEL + t)
    p_ptr = tl.load(PP + t).to(tl.pointer_type(tl.float32))
    g_ptr = tl.load(GP + t).to(tl.pointer_type(tl.float32))
    m_ptr = tl.load(MP + t).to(tl.pointer_type(tl.float32))
    v_ptr = tl.load(VP + t).to(tl.pointer_type(tl.float32))
    s_ptr = tl.load(SP + t).to(tl.pointer_type(tl.float32))
    lr = tl.load(HP + 0)
    b1 = tl.load(HP + 1)
    b2 = tl.load(HP + 2)
    eps = tl.load(HP + 3)
    lr_wd = tl.load(HP + 4)   # lr * weight_decay, in double, as `lr * weight_decay * param` multiplies left to right
    omb1 = tl.load(HP + 5)    # 1 - beta1 in double
    omb2 = tl.load(HP + 6)
    step = tl.load(s_ptr).to(tl.float64)
    bc1 = (1.0 - libdevice.pow(b1, step)).to(tl.float32)
    bc2s = libdevice.sqrt(1.0 - libdevice.pow(b2, step)).to(tl.float32)   # double sqrt (IEEE)
    step_size = (lr / bc1.to(tl.float64)).to(tl.float32)

    offs = off + tl.arange(0, BLOCK)
    mask = offs < n
    p = tl.load(p_ptr + offs, mask=mask, other=0.0)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0)
    m = tl.load(m_ptr + offs, mask=mask, other=0.0)
    v = tl.load(v_ptr + offs, mask=mask, other=0.0)
    if HAS_COEF:
        g = g * tl.load(COEF)
    if HAS_WD:   # param -= lr * weight_decay * param (double), contracted: fma(-(lr * wd), param, param)
        pd = p.to(tl.float64)
        p = tl.fma(-lr_wd, pd, pd).to(tl.float32)
    gd = g.to(tl.float64)
    md = m.to(tl.float64)
    vd = v.to(tl.float64)
    if FMA_M == 0:
        m = tl.fma(b1, md, omb1 * gd).to(tl.float32)
    elif FMA_M == 1:
        m = tl.fma(omb1, gd, b1 * md).to(tl.float32)
    else:
        m = (b1 * md + omb1 * gd).to(tl.float32)
    if FMA_V == 0:
        v = tl.fma(b2, vd, (omb2 * gd) * gd).to(tl.float32)
    elif FMA_V == 1:
        v = tl.fma(omb2 * gd, gd, b2 * vd).to(tl.float32)
    else:
        v = (b2 * vd + (omb2 * gd) * gd).to(tl.float32)
    denom = (tl.div_rn(tl.sqrt_rn(v), bc2s).to(tl.float64) + eps).to(tl.float32)
    p = p - tl.div_rn(step_size * m, denom)
    tl.store(p_ptr + offs, p, mask=mask)
    tl.store(m_ptr + offs, m, mask=mask)
    tl.store(v_ptr + offs, v, mask=mask)
    if HAS_EMA:
        e_ptr = tl.load(EP + t).to(tl.pointer_type(tl.float32))
        e = tl.load(e_ptr + offs, mask=mask, other=0.0)
        if FMA_E == 0:
            e = tl.fma(ema_w, p - e, e)
        else:
            e = e + ema_w * (p - e)
        tl.store(e_ptr + offs, e, mask=mask)


# the variant that equals torch 2.7.1 (nvcc -fmad=true) bit for bit on H100; set by the test, see the docstring
FMA_M, FMA_V, FMA_E = 0, 0, 0


def _local(ts):
    from torch.distributed.tensor import DTensor
    return [t._local_tensor if isinstance(t, DTensor) else t for t in ts]


class _Tables:
    """Device tables of one parameter group: per tensor its numel and the addresses of param, grad, exp_avg, exp_avg_sq,
    step and EMA; per program its tensor and offset. Rebuilt when an address changes (FSDP2 hands new gradient tensors)."""

    def __init__(self):
        self.key = None

    def get(self, lists, dev):
        key = tuple(t.data_ptr() for ts in lists for t in ts)
        if key != self.key:
            n = [t.numel() for t in lists[0]]
            for k, ts in enumerate(lists[1:], 1):   # list 4 holds the step counts (one element each)
                assert [t.numel() for t in ts] == (n if k != 4 else [1] * len(n))
            for ts in lists:
                for t in ts:
                    assert t.dtype == torch.float32 and t.is_contiguous() and t.device == dev, (t.dtype, t.shape)
            keep = [i for i, k in enumerate(n) if k > 0]   # an empty shard gets no program
            ptr = torch.tensor([[ts[i].data_ptr() for i in keep] for ts in lists], dtype=torch.int64)
            nb = [-(-n[i] // BLOCK) for i in keep]
            tid = torch.repeat_interleave(torch.arange(len(keep), dtype=torch.int32), torch.tensor(nb))
            toff = torch.cat([torch.arange(b, dtype=torch.int64) * BLOCK for b in nb]) if nb else torch.zeros(0, dtype=torch.int64)
            self.ptr = ptr.pin_memory().to(dev, non_blocking=True)
            self.numel = torch.tensor([n[i] for i in keep], dtype=torch.int64).pin_memory().to(dev, non_blocking=True)
            self.tid = tid.pin_memory().to(dev, non_blocking=True)
            self.toff = toff.pin_memory().to(dev, non_blocking=True)
            self.programs = int(tid.numel())
            self.key = key
            self.rebuilds = getattr(self, "rebuilds", 0) + 1
        return self


class FusedAdamWEMA(torch.optim.AdamW):
    """torch.optim.AdamW(fused=True) whose step runs _adamw_ema_kernel. Before a step, set `clip_coef` (the clamped clip
    coefficient tensor of clip_grad_norm_, used for that step only) and, for the fused EMA, `ema_of` ({param: EMA tensor})
    and `ema_weight` (1 - decay)."""

    @classmethod
    def like(cls, opt, param_groups):
        d = opt.defaults
        return cls(param_groups, lr=d["lr"], betas=d["betas"], eps=d["eps"], weight_decay=d["weight_decay"],
                   amsgrad=d["amsgrad"], maximize=d["maximize"], foreach=d["foreach"], capturable=d["capturable"],
                   differentiable=d["differentiable"], fused=d["fused"])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_coef = None
        self.ema_of = None
        self.ema_weight = None
        self._tables = {}
        self._hp = {}

    def _hyper(self, group, dev):
        lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
        assert not torch.is_tensor(lr), "a tensor lr is not supported"
        key = (lr, b1, b2, eps, wd)
        hp = self._hp.get(id(group))
        if hp is None or hp[0] != key:
            t = torch.tensor([lr, b1, b2, eps, lr * wd, 1 - b1, 1 - b2], dtype=torch.float64)
            hp = (key, t.pin_memory().to(dev, non_blocking=True))
            self._hp[id(group)] = hp
        return hp[1]

    @_use_grad_for_differentiable
    def step(self, closure=None):
        self._cuda_graph_capture_health_check()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        coef, self.clip_coef = self.clip_coef, None
        for group in self.param_groups:
            params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, steps = [], [], [], [], [], []
            self._init_group(group, params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, steps)
            if not params:
                continue
            assert group["fused"] and not group["amsgrad"] and not group["maximize"] and not group["differentiable"]
            assert getattr(self, "grad_scale", None) is None and getattr(self, "found_inf", None) is None
            torch._foreach_add_(steps, 1)   # as torch's _fused_adam
            lists = [_local(params), _local(grads), _local(exp_avgs), _local(exp_avg_sqs), steps]
            has_ema = self.ema_of is not None
            if has_ema:
                lists.append(_local([self.ema_of[p] for p in params]))
            dev = lists[0][0].device
            tb = self._tables.setdefault(id(group), _Tables()).get(lists, dev)
            if tb.programs == 0:
                continue
            hp = self._hyper(group, dev)
            co = coef if coef is not None else hp   # a placeholder pointer when there is no clip
            if coef is not None:
                assert coef.dtype == torch.float32 and coef.numel() == 1 and coef.device == dev
            _adamw_ema_kernel[(tb.programs,)](
                tb.ptr[0], tb.ptr[1], tb.ptr[2], tb.ptr[3], tb.ptr[4], tb.ptr[5] if has_ema else tb.ptr[0],
                tb.numel, tb.tid, tb.toff, co, hp, float(self.ema_weight) if has_ema else 0.0,
                HAS_COEF=coef is not None, HAS_EMA=has_ema, HAS_WD=group["weight_decay"] != 0,
                FMA_M=FMA_M, FMA_V=FMA_V, FMA_E=FMA_E, BLOCK=BLOCK, num_warps=8, enable_fp_fusion=False)
        return loss
