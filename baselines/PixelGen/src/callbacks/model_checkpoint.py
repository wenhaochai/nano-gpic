# import os.path
# from typing import Optional, Dict, Any

# import lightning.pytorch as pl
# from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

# commented this out
# class CheckpointHook(ModelCheckpoint):
#     """Save checkpoint with only the incremental part of the model"""
#     def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
#         self.dirpath = trainer.default_root_dir
#         self.exception_ckpt_path = os.path.join(self.dirpath, "on_exception.pt")
#         pl_module.strict_loading = False

#     def on_save_checkpoint(
#             self, trainer: "pl.Trainer",
#             pl_module: "pl.LightningModule",
#             checkpoint: Dict[str, Any]
#     ) -> None:
#         del checkpoint["callbacks"]

#     # def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
#     #     if not "debug" in self.exception_ckpt_path:
#     #         trainer.save_checkpoint(self.exception_ckpt_path)


import os.path
from pathlib import Path
from typing import Dict, Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

class CheckpointHook(ModelCheckpoint):
    """Save optimizer states only for last.ckpt."""

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

        self._saving_last_ckpt = False

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        self._saving_last_ckpt = Path(filepath).name == "last.ckpt"

        try:
            super()._save_checkpoint(trainer, filepath)
        finally:
            self._saving_last_ckpt = False

    def on_save_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ) -> None:
        checkpoint.pop("callbacks", None)

        if not self._saving_last_ckpt:
            print("Removing optimizer states and lr schedulers from checkpoint since it's not last.ckpt")
            print(f"{checkpoint.keys()}")
            checkpoint.pop("optimizer_states", None)
            checkpoint.pop("lr_schedulers", None)
            checkpoint.pop("loops", None)