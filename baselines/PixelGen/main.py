import os
import sys
import torch
import time
from typing import Any, Union

from src.utils.patch_bugs import *
from lightning import Trainer, LightningModule

from src.lightning_data_gpic import DataModule
from src.lightning_model_gpic import LightningModel
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, SaveConfigCallback

import logging
logger = logging.getLogger("lightning.pytorch")
# log_path = os.path.join( f"log.txt")
# logger.addHandler(logging.FileHandler(log_path))

class ReWriteRootSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        stamp = time.strftime('%y%m%d%H%M')
        file_path = os.path.join(trainer.default_root_dir, f"config-{stage}-{stamp}.yaml")
        self.parser.save(
            self.config, file_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
        )


class ReWriteRootDirCli(LightningCLI):
    def before_instantiate_classes(self) -> None:
        super().before_instantiate_classes()
        config_trainer = self._get(self.config, "trainer", default={})

        # predict path & logger check
        if self.subcommand == "predict":
            config_trainer.logger = None

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        class TagsClass:
            def __init__(self, exp:str):
                ...
        parser.add_class_arguments(TagsClass, nested_key="tags")

    def add_default_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_default_arguments_to_parser(parser)
        parser.add_argument("--torch_hub_dir", type=str, default=None, help=("torch hub dir"),)
        parser.add_argument("--huggingface_cache_dir", type=str, default=None, help=("huggingface hub dir"),)

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        config_trainer = self._get(self.config_init, "trainer", default={})
        default_root_dir = config_trainer.get("default_root_dir", None)

        if default_root_dir is None:
            default_root_dir = os.path.join(os.getcwd(), "workdirs")

        dirname = ""
        for v, k in self._get(self.config, "tags", default={}).items():
            dirname += f"{v}_{k}"
        default_root_dir = os.path.join(default_root_dir, dirname)
        is_resume = self._get(self.config_init, "ckpt_path", default=None)
        # if os.path.exists(default_root_dir) and "debug" not in default_root_dir:
        #     if os.listdir(default_root_dir) and self.subcommand != "predict" and not is_resume:
        #         raise FileExistsError(f"{default_root_dir} already exists")

        config_trainer.default_root_dir = default_root_dir
        trainer = super().instantiate_trainer(**kwargs)
        if trainer.is_global_zero:
            os.makedirs(default_root_dir, exist_ok=True)
        return trainer

    def instantiate_classes(self) -> None:
        torch_hub_dir = self._get(self.config, "torch_hub_dir")
        huggingface_cache_dir = self._get(self.config, "huggingface_cache_dir")
        if huggingface_cache_dir is not None:
            os.environ["HUGGINGFACE_HUB_CACHE"] = huggingface_cache_dir
        if torch_hub_dir is not None:
            os.environ["TORCH_HOME"] = torch_hub_dir
            torch.hub.set_dir(torch_hub_dir)
        super().instantiate_classes()

    def before_fit(self) -> None:
        # Launch lever A1 (gpic-28 2026-11-09). jobs/train_ladder.sh starts the GPU health gate (tools/gpu_health.py) and
        # this process together: the imports and the model's construction use no GPU (0 MiB on every GPU for 40 s after
        # the gate in the w2441 profile), so the gate's 15 s no longer come before them. This is the last point before
        # the first CUDA call, and fit waits here for the gate's verdict. GPIC_GATE_RC: the file that gets the gate's
        # exit status; GPIC_GATE_PID: the shell that writes it. A failed gate, a gate that ended without a verdict or
        # no verdict in GPIC_GATE_WAIT_S ends the launch with 98, the blocking gate's code; nothing has started yet
        # (no rank, no data worker, no wandb run). The ranks that Lightning starts later find the file in place.
        rc_file = os.environ.get("GPIC_GATE_RC")
        if not rc_file:
            return

        def read():
            try:
                with open(rc_file) as f:
                    return int(f.read().strip())
            except FileNotFoundError:
                return None
            except ValueError:
                return -1

        def gate_gone():   # the gate's shell has ended (a zombie until its parent's wait) or is not there
            try:
                with open(f"/proc/{pid}/stat") as f:
                    return f.read().rsplit(")", 1)[1].split()[0] == "Z"
            except FileNotFoundError:
                return True
            except Exception:
                return False

        t0, pid = time.time(), int(os.environ.get("GPIC_GATE_PID") or 0)
        limit = float(os.environ.get("GPIC_GATE_WAIT_S") or 300)
        rc = read()
        while rc is None and time.time() - t0 < limit:
            if pid and gate_gone():   # its verdict is in the file now, or it never comes
                rc = read()
                break
            time.sleep(0.05)
            rc = read()
        first = int(os.environ.get("LOCAL_RANK") or 0) == 0
        if rc == 0:
            if first:
                print(f"[gate] healthy; fit waited {time.time() - t0:.1f} s for the verdict", flush=True)
            return
        if first:
            print(f"[gate] {'no verdict' if rc is None else f'exit status {rc}'} after {time.time() - t0:.1f} s: no fit", flush=True)
        sys.exit(98)

def leave_after_failure(exc, code=1):
    """GPIC_FAIL_FAST (on by default, 0 = off; 2026-11-10, w2503). An exception that escapes fit used to end the process
    through the interpreter's exit hooks, and one of them waits for the other ranks: Lightning registers
    _destroy_dist_connection (lightning/fabric/utilities/distributed.py:287) for NCCL, and torch's destroy_process_group
    -> ProcessGroupNCCL::shutdown finalizes the communicator and waits for it. In w2503 rank 0 failed in wandb.init
    (trainer/call.py:102, before the "pre_setup" barrier at call.py:105) while ranks 1-7 sat in that barrier; rank 0 printed
    its traceback, wandb's own exit hook ended at 18:43:03, and then rank 0 stayed alive with GPU 0 idle until it was
    cancelled at 18:51:51 (its NCCL watchdog was still running at 18:51:30, so shutdown had not got past its first stage).
    Here: the traceback, then the other exit hooks (wandb's teardown, the checkpoint writer's wait, the data workers) in a
    thread for at most GPIC_FAIL_EXIT_HOOKS_S (30) seconds (wandb's took 1.4 s in w2503), without the NCCL destroy, then os._exit with the status. The
    other ranks wait in a collective for this one: jobs/train_ladder.sh ends them (by GPIC_LAUNCH_ID)."""
    import atexit
    import threading
    import traceback
    try:
        sys.excepthook(type(exc), exc, exc.__traceback__)   # torch's hook prefixes the rank once the group exists
    except Exception:
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    for s in (sys.stdout, sys.stderr):
        try:
            s.flush()
        except Exception:
            pass
    try:
        from lightning.fabric.utilities.distributed import _destroy_dist_connection
        atexit.unregister(_destroy_dist_connection)
    except Exception:
        pass
    limit = float(os.environ.get("GPIC_FAIL_EXIT_HOOKS_S") or 30)
    t0 = time.time()
    hooks = threading.Thread(target=atexit._run_exitfuncs, name="exit-hooks", daemon=True)
    hooks.start()
    hooks.join(limit)
    note = f"; exit hooks still running after {limit:.0f} s" if hooks.is_alive() else ""
    print(f"[main] fit failed ({type(exc).__name__}); leaving with status {code} {time.time() - t0:.1f} s after the "
          f"traceback{note}", file=sys.stderr, flush=True)
    try:
        sys.stdout.flush()
    except Exception:
        pass
    os._exit(code)


if __name__ == "__main__":

    try:
        cli = ReWriteRootDirCli(LightningModel, DataModule,
                                auto_configure_optimizers=False,
                                save_config_callback=ReWriteRootSaveConfigCallback,
                                save_config_kwargs={"overwrite": True})
    except SystemExit:   # a status chosen on purpose (the gate's 98, an argument error, Lightning's own exits): as before
        raise
    except BaseException as e:
        if os.environ.get("GPIC_FAIL_FAST", "1") != "1":
            raise
        leave_after_failure(e)
    train_dataset = getattr(cli.datamodule, "train_dataset", None)
    if hasattr(train_dataset, "stop_workers"):
        train_dataset.stop_workers()
