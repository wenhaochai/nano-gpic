import os.path
from pathlib import Path
from typing import Dict, Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint


class CheckpointHook(ModelCheckpoint):
    """Full state only in last*.ckpt (model, optimizer, LR schedulers, loops, callbacks); every other file keeps the
    model only.

    Lightning calls on_save_checkpoint of every callback for every save, so the decision follows the file being
    written, held in a class attribute shared by all instances (the previous per-instance flag let a second hook strip
    last.ckpt: resume failed with "checkpoint contains only the model"). The callback state stays in last*.ckpt, so a
    resumed run's ModelCheckpoint knows last.ckpt is its own and overwrites it instead of writing last-v1.ckpt
    (Muon review B1, B4, 2026-11-02)."""

    _current_target = None   # path of the checkpoint being written by any CheckpointHook

    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: str,
    ) -> None:
        super().setup(trainer, pl_module, stage)

        self.dirpath = trainer.default_root_dir
        self.exception_ckpt_path = os.path.join(self.dirpath, "on_exception.pt")
        pl_module.strict_loading = False

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        CheckpointHook._current_target = filepath
        try:
            super()._save_checkpoint(trainer, filepath)
        finally:
            CheckpointHook._current_target = None

    def on_save_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ) -> None:
        target = CheckpointHook._current_target
        if target is None or Path(target).name.startswith("last"):
            return   # last*.ckpt, or a save that no CheckpointHook started: keep everything
        for key in ("callbacks", "optimizer_states", "lr_schedulers", "loops"):
            checkpoint.pop(key, None)
