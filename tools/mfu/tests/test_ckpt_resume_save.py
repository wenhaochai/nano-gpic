#!/usr/bin/env python3
"""CPU test of the two checkpoint hooks of jobs/train_ladder.sh across launches (2026-11-09): a tiny model, the hooks
built from the launcher's own CB line, each launch resumed from the newest last*.ckpt as the launcher does.
Launch 1 trains 8 steps (ms every 2, last every 1). Then:
  old      the code before this change (version counter on the last hook, Lightning's own on_train_start), an extension
           with 2 micro-batches a step: it must show the fault (ms-step=000008-v1.ckpt and last-v1.ckpt, both written
           at step 8 in the middle of an accumulation window), or this test tests nothing
  ext      an extension to 16 steps that changes both intervals (ms 4, last 2), with 1 and with 2 micro-batches a step
  same     a resume with the intervals unchanged, 2 micro-batches a step
  branch   a new run directory that holds only a copy of the parent's last.ckpt, 2 micro-batches a step
  moved    a run that the old code left with a live last-v1.ckpt beside a stale last.ckpt, continued by the new code
In every new case: no save at the step the launch resumed from, every save at the end of an accumulation window, no
-vN file written, last.ckpt = the newest last*.ckpt = the full state of the last step, ms files model only.
usage: python test_ckpt_resume_save.py <scratch dir>"""
import glob, json, os, shutil, sys, tempfile
import torch, lightning.pytorch as pl
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, os.path.join(ROOT, "baselines", "PixelGen"))
from src.callbacks.model_checkpoint import CheckpointHook   # noqa: E402

line = next(l for l in open(os.path.join(ROOT, "jobs", "train_ladder.sh")) if l.startswith("CB='["))
for shell, value in (("'$MS'", "0"), ("'$LAST'", "0"), ("'$RUN/none_refcount.txt'", "x"), ("'$RUN'", "x"), ("'$BENCH_CB'", "")):
    assert shell in line, shell
    line = line.replace(shell, value)
ARGS = [c["init_args"] for c in json.loads(line[len("CB='"):].rstrip("\n").rstrip("'")) if c["class_path"].endswith(".CheckpointHook")]
assert len(ARGS) == 2 and ARGS[0]["save_last"] is False and ARGS[1]["save_last"] is True, ARGS

SAVES = []
_save, _start = CheckpointHook._save_checkpoint, CheckpointHook.on_train_start


def _logged(self, trainer, filepath):   # (file, step, micro-batches done): the class keeps its name, so its state key too
    SAVES.append((os.path.basename(filepath), trainer.global_step, trainer.fit_loop.epoch_loop.batch_progress.current.processed))
    _save(self, trainer, filepath)


CheckpointHook._save_checkpoint = _logged


class DS(torch.utils.data.IterableDataset):
    def __iter__(self):
        g = torch.Generator().manual_seed(0)
        while True:
            yield torch.randn(4, generator=g)


class M(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(4, 1)

    def training_step(self, b, i):
        return self.l(b).pow(2).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def newest(root):
    return max(glob.glob(f"{root}/last*.ckpt"), key=lambda f: os.stat(f).st_mtime_ns, default=None)


def launch(root, steps, ms, last, acc, old=False):
    a = [dict(ARGS[0], every_n_train_steps=ms), dict(ARGS[1], every_n_train_steps=last)]
    if old:
        a[1].pop("enable_version_counter")
    CheckpointHook.on_train_start = ModelCheckpoint.on_train_start if old else _start
    del SAVES[:]
    t = pl.Trainer(max_steps=steps, callbacks=[CheckpointHook(**a[0]), CheckpointHook(**a[1])], default_root_dir=root, logger=False,
                   enable_progress_bar=False, enable_model_summary=False, accelerator="cpu", devices=1, accumulate_grad_batches=acc)
    t.fit(M(), torch.utils.data.DataLoader(DS(), batch_size=2), ckpt_path=newest(root))
    return [s[:2] for s in SAVES], [s for s in SAVES if s[2] % acc]


def files(root):
    return sorted(os.path.basename(f) for f in glob.glob(f"{root}/*.ckpt"))


def state(root, step):
    """last.ckpt is the newest last file, holds the full state of <step> and names itself; the ms file of <step> holds the model only"""
    f = newest(root)
    assert os.path.basename(f) == "last.ckpt", f
    ck = torch.load(f, map_location="cpu", weights_only=False)
    assert ck["global_step"] == step and all(k in ck for k in ("optimizer_states", "lr_schedulers", "loops", "callbacks")), (ck["global_step"], list(ck))
    assert {os.path.basename(v["last_model_path"]) for v in ck["callbacks"].values() if v.get("last_model_path")} == {"last.ckpt"}
    ms = torch.load(f"{root}/ms-step={step:06d}.ckpt", map_location="cpu", weights_only=False)
    assert ms["global_step"] == step and not any(k in ms for k in ("optimizer_states", "loops", "callbacks")), list(ms)
    assert all(torch.equal(ms["state_dict"][k], ck["state_dict"][k]) for k in ck["state_dict"])


def first(top, name, acc):
    root = f"{top}/{name}"
    os.makedirs(root)
    saves, mid = launch(root, 8, 2, 1, acc)
    assert not mid and files(root) == ["last.ckpt"] + [f"ms-step={s:06d}.ckpt" for s in (2, 4, 6, 8)], (name, mid, files(root))
    return root


EXT = [("last.ckpt", 10), ("ms-step=000012.ckpt", 12), ("last.ckpt", 12), ("last.ckpt", 14), ("ms-step=000016.ckpt", 16), ("last.ckpt", 16)]
top = tempfile.mkdtemp(dir=sys.argv[1], prefix="ckpt_resume_save_")
try:
    root = first(top, "old", 2)
    saves, mid = launch(root, 16, 4, 2, 2, old=True)
    assert saves[:2] == [("ms-step=000008-v1.ckpt", 8), ("last-v1.ckpt", 8)] and len(mid) == 2 and mid[0][1] == mid[1][1] == 8, (saves, mid)
    assert torch.load(f"{root}/last.ckpt", map_location="cpu", weights_only=False)["global_step"] == 8   # the stale file
    print(f"ok  old code, extension, 2 micro-batches a step: the fault ({saves[0][0]} and {saves[1][0]} at step 8, written after micro-batch {mid[0][2] % 2} of the 2 of step 9)")

    for acc in (1, 2):
        root = first(top, f"ext{acc}", acc)
        saves, mid = launch(root, 16, 4, 2, acc)
        assert saves == EXT and not mid, (saves, mid)
        assert files(root) == ["last.ckpt"] + [f"ms-step={s:06d}.ckpt" for s in (2, 4, 6, 8, 12, 16)], files(root)
        state(root, 16)
        print(f"ok  extension (ms 2 -> 4, last 1 -> 2), {acc} micro-batch(es) a step: first save at step {saves[0][1]}, no -vN file, last.ckpt at step 16")

    root = first(top, "same", 2)
    saves, mid = launch(root, 16, 2, 1, 2)
    want = [x for s in range(9, 17) for x in ([(f"ms-step={s:06d}.ckpt", s)] if s % 2 == 0 else []) + [("last.ckpt", s)]]
    assert saves == want and not mid, (saves, mid)
    assert files(root) == ["last.ckpt"] + [f"ms-step={s:06d}.ckpt" for s in range(2, 17, 2)], files(root)
    state(root, 16)
    print("ok  resume with the intervals unchanged, 2 micro-batches a step: first save at step 9 (before: step 8 written again)")

    parent = first(top, "parent", 2)
    root = f"{top}/branch"
    os.makedirs(root)
    shutil.copy2(f"{parent}/last.ckpt", f"{root}/last.ckpt")
    before = {f: os.stat(f"{parent}/{f}").st_mtime_ns for f in files(parent)}
    saves, mid = launch(root, 16, 4, 2, 2)
    assert saves == EXT and not mid, (saves, mid)
    assert files(root) == ["last.ckpt", "ms-step=000012.ckpt", "ms-step=000016.ckpt"], files(root)
    assert before == {f: os.stat(f"{parent}/{f}").st_mtime_ns for f in files(parent)}, "the branch wrote into its parent's directory"
    state(root, 16)
    print("ok  branch from a copied last.ckpt: the copy is written over at step 10, no last-v1.ckpt, the parent's files untouched")

    root = first(top, "moved", 2)
    saves, mid = launch(root, 12, 4, 2, 2, old=True)
    assert os.path.basename(newest(root)) == "last-v1.ckpt" and saves[-1] == ("last-v1.ckpt", 12), saves
    saves, mid = launch(root, 16, 4, 2, 2)
    assert saves == [("last.ckpt", 14), ("ms-step=000016.ckpt", 16), ("last.ckpt", 16)] and not mid, (saves, mid)
    state(root, 16)
    left = [f for f in files(root) if "-v" in f]
    print(f"ok  a run with a live last-v1.ckpt (old code), continued: resumed at 12, last.ckpt is the newest again at step 16; old files left: {left}")
    print("TEST_CKPT_RESUME_SAVE PASS")
finally:
    shutil.rmtree(top, ignore_errors=True)
