from typing import Any, Dict

import torch
import torch.nn as nn
import threading
import lightning.pytorch as pl
from lightning.pytorch import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

from src.utils.copy import swap_tensors

class SimpleEMA(Callback):
    def __init__(self,
                 decay: float = 0.9999,
                 every_n_steps: int = 1,
                 ):
        super().__init__()
        self.decay = decay
        self.every_n_steps = every_n_steps
        self._stream = None   # perf C3: created on the parameters' own device (was GPU 0 on every rank)
        self.previous_step = 0
        self.fused = False    # MFU lever H (GPIC_MFU_FUSED_OPT=2): the optimizer's kernel updates the EMA

    def setup_models(self, net: nn.Module, ema_net: nn.Module):
        self.net_params = list(net.parameters())
        self.ema_params = list(ema_net.parameters())
        # MFU lever B (FSDP2): sharded (DTensor) and replicated parameters go to separate foreach calls; the EMA module is
        # sharded like the denoiser, so each rank updates its own shard (src/utils/fsdp_setup.py)
        from torch.distributed.tensor import DTensor
        dt = [isinstance(p, DTensor) for p in self.net_params]
        assert dt == [isinstance(p, DTensor) for p in self.ema_params], "EMA and denoiser sharded alike"
        self._groups = [([e for e, d in zip(self.ema_params, dt) if d == k], [n for n, d in zip(self.net_params, dt) if d == k])
                        for k in (True, False)]
        self._groups = [g for g in self._groups if g[0]]
        # MFU lever B2 (GPIC_MFU_LOCAL_OPT=1): the lerp on the DTensors' local tensors, taken at every step (FSDP2 may
        # re-point a shard's local tensor at its lazy init): the same kernel on the same data without DTensor's dispatch
        import os
        self._local = os.environ.get("GPIC_MFU_LOCAL_OPT") == "1"

    def ema_step(self):
        @torch.no_grad()
        def ema_update(ema_model_tuple, current_model_tuple, decay):
            # perf C3: one pass, ema += (1 - decay) * (p - ema) == decay * ema + (1 - decay) * p
            torch._foreach_lerp_(ema_model_tuple, current_model_tuple, 1.0 - decay)

        if self._stream is None:
            self._stream = torch.cuda.Stream(device=self.ema_params[0].device)
        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            for ema_g, net_g in self._groups:
                if self._local:
                    from src.utils.fsdp_setup import _local
                    ema_g, net_g = _local(ema_g), _local(net_g)
                ema_update(ema_g, net_g, self.decay)
        assert self.ema_params[0].dtype == torch.float32

    def _join(self):
        """The default stream waits for the EMA update: the next optimizer step must not change the parameters the
        EMA is still reading, and a checkpoint must see the finished EMA (perf C3; both raced before)."""
        if self._stream is not None:
            torch.cuda.current_stream().wait_stream(self._stream)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        self._join()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        self._join()
        torch.cuda.current_stream().synchronize()

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if trainer.global_step == self.previous_step:
            return
        self.previous_step = trainer.global_step
        if self.fused:
            return
        if trainer.global_step % self.every_n_steps == 0:
            self.ema_step()


    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "every_n_steps": self.every_n_steps,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.decay = state_dict["decay"]
        self.every_n_steps = state_dict["every_n_steps"]
