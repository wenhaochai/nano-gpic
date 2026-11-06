import os
import time

import torch
import torch.nn as nn


class BaseTrainer(nn.Module):
    def __init__(self,
                 null_condition_p=0.1,
        ):
        super(BaseTrainer, self).__init__()
        self.null_condition_p = null_condition_p

    def preproprocess(self, x, condition, uncondition, metadata):
        bsz = x.shape[0]
        if self.null_condition_p > 0 and isinstance(metadata, dict) and metadata.get("y_pack") is not None:
            raise NotImplementedError("MFU lever E (packed captions) has no caption dropout; the ladder uses none")
        if self.null_condition_p > 0:
            mask = torch.rand((bsz), device=condition.device) < self.null_condition_p
            if isinstance(metadata, dict) and metadata.get("y_mask") is not None:   # ladder option B (masked text)
                ym = metadata["y_mask"]
                null_row = torch.zeros_like(ym[:1]); null_row[:, 0] = True       # the null caption: one token
                metadata["y_mask"] = torch.where(mask[:, None], null_row, ym)
            mask = mask.view(-1, *([1] * (len(condition.shape) - 1))).to(condition.dtype)
            condition = condition*(1-mask)  + uncondition*mask
        return x, condition, metadata

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        raise NotImplementedError

    def __call__(self, net, ema_net, solver, x, condition, uncondition, metadata=None):
        if os.environ.get("GPIC_PROF_DEN_COMPILE_MODE") == "reduce-overhead":   # profiling switch, unset in production
            torch.compiler.cudagraph_mark_step_begin()
        x, condition, metadata = self.preproprocess(x, condition, uncondition, metadata)
        return self._impl_trainstep(net, ema_net, solver, x, condition, metadata)

