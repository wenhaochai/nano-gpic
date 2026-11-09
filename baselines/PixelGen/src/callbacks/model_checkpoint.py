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
    (Muon review B1, B4, 2026-11-02).

    That ownership is lost when the launch starts from a parent run's copied file (the stored dirpath differs) or
    changes every_n_train_steps (the hook's state key differs), so jobs/train_ladder.sh gives the last hook
    enable_version_counter false: last.ckpt is always the live state (before, such a run wrote last-v1.ckpt and left
    a stale last.ckpt beside it). And no hook saves at the step a launch resumed from (on_train_start): Lightning does
    not keep its last saved step in the callback state, and with accumulation > 1 the first micro-batch ends with
    global_step unchanged, so it wrote that step again, in the middle of an accumulation window: last.ckpt once more
    at every resume, and ms-step=<step>-v1.ckpt, a copy of the ms file, when the ms interval had changed (width 1792,
    2 micro-batches a step: 3 copies of 13.4 GiB in muon_full, which stopped two held-out jobs; 2026-11-09).

    Rank 0 prints one line per save, "[ckpt] step <s> (<n> micro-batches): <file>": n = s x accumulation at the end
    of a window."""

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

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_start(trainer, pl_module)
        self._last_global_step_saved = trainer.global_step   # 0, or the step of the file this launch resumed from

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        CheckpointHook._current_target = filepath
        try:
            super()._save_checkpoint(trainer, filepath)
        finally:
            CheckpointHook._current_target = None
        if trainer.is_global_zero:   # one line per save: a save in the middle of an accumulation window shows in the log
            done = trainer.fit_loop.epoch_loop.batch_progress.total.processed
            print(f"[ckpt] step {trainer.global_step} ({done} micro-batches): {os.path.basename(filepath)}", flush=True)

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
