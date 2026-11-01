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

    def setup_models(self, net: nn.Module, ema_net: nn.Module):
        self.net_params = list(net.parameters())
        self.ema_params = list(ema_net.parameters())

    def ema_step(self):
        @torch.no_grad()
        def ema_update(ema_model_tuple, current_model_tuple, decay):
            # perf C3: one pass, ema += (1 - decay) * (p - ema) == decay * ema + (1 - decay) * p
            torch._foreach_lerp_(ema_model_tuple, current_model_tuple, 1.0 - decay)

        if self._stream is None:
            self._stream = torch.cuda.Stream(device=self.ema_params[0].device)
        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            ema_update(self.ema_params, self.net_params, self.decay)
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
