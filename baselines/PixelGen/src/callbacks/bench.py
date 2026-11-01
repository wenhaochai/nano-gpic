"""Benchmark timer for the training loop (who-ate-my-flops campaign).

Records, per rank, into $BENCH_OUT:
  steps_rank<r>.json : per-step GPU time (CUDA events around the whole training batch), host wall time per step,
                       data wait (host time between the end of one batch and the start of the next),
                       peak allocated / reserved GPU memory and CPU RSS per step, and startup times.
  trace/rank<r>.json : torch.profiler chrome trace for steps [a, b) when BENCH_PROFILE="a:b".
It changes nothing that the job computes.
"""
import json
import os
import time

import lightning.pytorch as pl
import torch
from lightning.pytorch import Callback

_T0 = float(os.environ.get("BENCH_T0", time.time()))  # process start (set by the launcher)


def _rss_gb():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
    except Exception:
        return float("nan")


class BenchTimer(Callback):
    def __init__(self):
        super().__init__()
        self.out = os.environ.get("BENCH_OUT", ".")
        prof = os.environ.get("BENCH_PROFILE", "")
        self.prof_range = tuple(int(v) for v in prof.split(":")) if prof else None
        self.prof = None
        self.rows = []
        self.t_fit_start = None
        self.t_last_end = None
        self.ev = None
        self.events = []

    def on_fit_start(self, trainer, pl_module):
        self.t_fit_start = time.time()

    def on_train_start(self, trainer, pl_module):
        self.t_train_start = time.time()
        torch.cuda.reset_peak_memory_stats()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.time()
        step = trainer.global_step
        if self.prof_range and step == self.prof_range[0] and self.prof is None:
            self.prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True, with_stack=False, profile_memory=True)
            self.prof.__enter__()
        self.cur = {"step": step,
                    "data_wait_s": (now - self.t_last_end) if self.t_last_end else (now - self.t_train_start),
                    "t_start": now - _T0}
        self.ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        self.ev[0].record()
        if self.prof is not None:
            self._rf = torch.profiler.record_function(f"ProfilerStep#{step}")
            self._rf.__enter__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.ev[1].record()
        if self.prof is not None:
            self._rf.__exit__(None, None, None)
        now = time.time()
        self.events.append(self.ev)
        self.cur.update({
            "host_s": now - _T0 - self.cur["t_start"],
            "t_end": now - _T0,
            "max_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
            "max_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
            "rss_gb": _rss_gb(),
        })
        self.rows.append(self.cur)
        self.t_last_end = now
        if self.prof is not None and trainer.global_step >= self.prof_range[1]:
            self.prof.__exit__(None, None, None)
            os.makedirs(os.path.join(self.out, "trace"), exist_ok=True)
            self.prof.export_chrome_trace(os.path.join(self.out, "trace", f"rank{trainer.global_rank}.json"))
            self.prof = None
            self.prof_range = None
        # No synchronize per step: CUDA events are read once at the end, so CPU run-ahead is preserved.

    def on_train_end(self, trainer, pl_module):
        torch.cuda.synchronize()
        for i, (row, (a, b)) in enumerate(zip(self.rows, self.events)):
            row["gpu_ms"] = a.elapsed_time(b)                      # first to last kernel of this batch's stream work
            if i + 1 < len(self.events):
                row["step_ms"] = a.elapsed_time(self.events[i + 1][0])   # start-to-start: the steady step time
        os.makedirs(self.out, exist_ok=True)
        r = trainer.global_rank
        summary = {
            "rank": r,
            "process_start_to_fit_start_s": self.t_fit_start - _T0 if self.t_fit_start else None,
            "fit_start_to_train_start_s": (self.t_train_start - self.t_fit_start) if self.t_fit_start else None,
            "train_start_to_first_step_end_s": (self.rows[0]["t_end"] + _T0 - self.t_train_start) if self.rows else None,
            "rows": self.rows,
        }
        with open(os.path.join(self.out, f"steps_rank{r}.json"), "w") as f:
            json.dump(summary, f)
