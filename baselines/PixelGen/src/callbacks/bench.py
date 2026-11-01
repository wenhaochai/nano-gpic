"""Benchmark timer and profiler hooks for the training loop (who-ate-my-flops campaign).

It changes nothing that the job computes. Environment switches (all off by default):
  BENCH_OUT=<dir>           where every output goes
  BENCH_PROFILE=a:b         torch.profiler over steps [a, b): shapes, FLOPs, Python stacks, memory;
                            writes trace/rank<r>.json and, on rank 0, op tables
  BENCH_CUDA_PROFILER=a:b   cudaProfilerStart/Stop around steps [a, b) (capture window for nsys / ncu)
  BENCH_MARKERS=1           stage ranges (record_function + NVTX): stage:fwd, stage:bwd, stage:opt, stage:between
                            (on automatically when BENCH_PROFILE or BENCH_CUDA_PROFILER is set)
Per rank it writes steps_rank<r>.json: per-step GPU span and start-to-start step time (CUDA events, read once at
the end so CPU run-ahead is kept), host time, data wait, peak memory, RSS, startup times, and compile statistics.
"""
import json
import os
import time

import torch
from lightning.pytorch import Callback

_T0 = float(os.environ.get("BENCH_T0", time.time()))  # process start (set by the launcher)


def _rss_gb():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
    except Exception:
        return float("nan")


def _range(env):
    v = os.environ.get(env, "")
    return tuple(int(x) for x in v.split(":")) if v else None


class _Marker:
    """One open stage range, mirrored as a profiler record_function and an NVTX range."""

    def __init__(self, name):
        self.rf = torch.profiler.record_function(name)
        self.rf.__enter__()
        torch.cuda.nvtx.range_push(name)

    def close(self):
        torch.cuda.nvtx.range_pop()
        self.rf.__exit__(None, None, None)


class BenchTimer(Callback):
    def __init__(self):
        super().__init__()
        self.out = os.environ.get("BENCH_OUT", ".")
        self.prof_range = _range("BENCH_PROFILE")
        self.cuprof_range = _range("BENCH_CUDA_PROFILER")
        self.markers_on = bool(os.environ.get("BENCH_MARKERS")) or bool(self.prof_range) or bool(self.cuprof_range)
        self.prof = None
        self.rows, self.events = [], []
        self.t_fit_start = self.t_last_end = None
        self.ev = None
        self.mark = None

    # ------------------------------------------------------------------ stage markers
    def _stage(self, name):
        if not self.markers_on:
            return
        if self.mark is not None:
            self.mark.close()
        self.mark = _Marker(name) if name else None

    def on_fit_start(self, trainer, pl_module):
        self.t_fit_start = time.time()

    def on_train_start(self, trainer, pl_module):
        self.t_train_start = time.time()
        torch.cuda.reset_peak_memory_stats()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.time()
        step = trainer.global_step
        self._stage(None)  # closes stage:between
        if self.cuprof_range and step == self.cuprof_range[0]:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
        if self.prof_range and step == self.prof_range[0] and self.prof is None:
            self.prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True, with_stack=True, with_flops=True, profile_memory=True)
            self.prof.__enter__()
        self.cur = {"step": step,
                    "data_wait_s": (now - self.t_last_end) if self.t_last_end else (now - self.t_train_start),
                    "t_start": now - _T0}
        self.ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        self.ev[0].record()
        if self.prof is not None:
            self._rf = torch.profiler.record_function(f"ProfilerStep#{step}")
            self._rf.__enter__()
        self._stage("stage:fwd")

    def on_before_backward(self, trainer, pl_module, loss):
        self._stage("stage:bwd")

    def on_after_backward(self, trainer, pl_module):
        self._stage("stage:after_bwd")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        self._stage("stage:opt")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._stage(None)
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
        step = trainer.global_step  # already incremented
        if self.prof is not None and step >= self.prof_range[1]:
            self.prof.__exit__(None, None, None)
            self._export_profile(trainer.global_rank)
            self.prof = None
            self.prof_range = None
        if self.cuprof_range and step >= self.cuprof_range[1]:
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            self.cuprof_range = None
        self._stage("stage:between")  # EMA callback, progress bar, logging, next-batch fetch

    def _export_profile(self, r):
        d = os.path.join(self.out, "trace")
        os.makedirs(d, exist_ok=True)
        self.prof.export_chrome_trace(os.path.join(d, f"rank{r}.json"))
        if r == 0:
            ka = self.prof.key_averages
            with open(os.path.join(d, "rank0_ops.txt"), "w") as f:
                f.write(ka().table(sort_by="self_cuda_time_total", row_limit=300, max_name_column_width=90))
            with open(os.path.join(d, "rank0_ops_by_shape.txt"), "w") as f:
                f.write(ka(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=400,
                                                             max_name_column_width=70, max_shapes_column_width=160))
            with open(os.path.join(d, "rank0_ops_by_stack.txt"), "w") as f:
                f.write(ka(group_by_stack_n=8).table(sort_by="self_cuda_time_total", row_limit=200,
                                                     max_name_column_width=70))

    def on_train_end(self, trainer, pl_module):
        self._stage(None)
        torch.cuda.synchronize()
        for i, (row, (a, b)) in enumerate(zip(self.rows, self.events)):
            row["gpu_ms"] = a.elapsed_time(b)                            # first to last kernel of this batch
            if i + 1 < len(self.events):
                row["step_ms"] = a.elapsed_time(self.events[i + 1][0])   # start-to-start: the steady step time
        os.makedirs(self.out, exist_ok=True)
        r = trainer.global_rank
        compile_stats = {}
        try:
            import torch._dynamo.utils as du
            compile_stats["compile_times"] = du.compile_times(repr="str", aggregate=True)
            compile_stats["counters"] = {k: dict(v) for k, v in du.counters.items()}
        except Exception as e:  # stats are best effort
            compile_stats["error"] = repr(e)
        summary = {
            "rank": r,
            "process_start_to_fit_start_s": self.t_fit_start - _T0 if self.t_fit_start else None,
            "fit_start_to_train_start_s": (self.t_train_start - self.t_fit_start) if self.t_fit_start else None,
            "train_start_to_first_step_end_s": (self.rows[0]["t_end"] + _T0 - self.t_train_start) if self.rows else None,
            "compile": compile_stats,
            "rows": self.rows,
        }
        with open(os.path.join(self.out, f"steps_rank{r}.json"), "w") as f:
            json.dump(summary, f, default=str)
