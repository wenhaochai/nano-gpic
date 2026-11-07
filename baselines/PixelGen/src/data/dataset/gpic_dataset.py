import os
import warnings
import numpy as np
import json, re, math
import webdataset as wds
import torch
from torchvision import transforms as T
import csv
from torch.utils.data import IterableDataset, Dataset

# -------------------------
# Transforms
# -------------------------

from .augmentation import center_crop_arr, random_crop_arr, random_crop_arr_region
global train_transform, val_transform

_train_crop = random_crop_arr_region if os.environ.get("GPIC_REGION_CROP") == "1" else random_crop_arr   # docs/text_cache.md; jobs/train_ladder.sh sets it
_UINT8 = os.environ.get("GPIC_MFU_UINT8") == "1"   # MFU lever D: grouped training batches carry uint8 HWC images
_PACK = os.environ.get("GPIC_MFU_PACK") == "1"     # MFU lever E: packed captions (src/utils/rebalance.py)
# MFU lever C: the region crop's two resize passes run on the GPU, bit for bit Pillow's (pil_exact.py, pil_exact_gpu.py);
# the workers decode, draw the crop and compute the coefficient tables. Needs the region crop (GPIC_REGION_CROP=1).
_GPU_RESIZE = os.environ.get("GPIC_MFU_GPU_RESIZE") == "1"
# tests only (MFU parity runs): the grouping RNG and the sample shuffle seeded from the shard seed (production seeds them from
# the PID and the clock), so two runs read the same batches and their per-step losses can be compared
_DET = os.environ.get("GPIC_MFU_DET_DATA") == "1"
assert not _GPU_RESIZE or os.environ.get("GPIC_REGION_CROP") == "1", "GPIC_MFU_GPU_RESIZE needs GPIC_REGION_CROP=1"
train_transform = T.Compose([
    T.Lambda(lambda pil_image: _train_crop(pil_image, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])
print(f"Defined train_transform function (from dataset.augmentation import {_train_crop.__name__}) :", train_transform)

val_transform = T.Compose([
        T.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
print("Defined val_transform function (from dataset.augmentation import center_crop_arr) :", val_transform)

# -------------------------
# Utils
# -------------------------

EMB_DIM = 2048                # the text encoder's output width
EMB_TOKEN_BYTES = EMB_DIM * 2   # one token of bf16 outputs
IMAGE_SUFFIXES = ("jpg", "jpeg", "png", "webp")   # webdataset lowers the suffixes
_DECODE_PIL = wds.imagehandler("pil")

def _rank_world():
    """(rank, world size): torch.distributed when it is up (also in a forked data worker), else the launcher's env.
    Lightning's launcher sets WORLD_SIZE but not RANK, so a world of several processes without torch.distributed and
    without RANK has no rank to give: every process would take rank 0 (review of ladder-full, M1), so that raises."""
    dist = torch.distributed
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1 and "RANK" not in os.environ:
        raise RuntimeError(f"WORLD_SIZE={world} but neither torch.distributed nor RANK gives this process its rank")
    return int(os.environ.get("RANK", 0)), world


class EpochShuffledShards(IterableDataset):
    """ladder: the shard list, reshuffled at every pass with Random(seed * 100003 + pass), and split by node here: rank r
    reads the shards r, r + world, r + 2 world, ... of the pass's permutation (what wds.split_by_node did after this
    stage, so the pipeline has no split_by_node). Every rank and worker holds its own copy and makes exactly one pass
    per epoch (persistent workers call iter() once per epoch), so in epoch e every copy uses the same permutation and
    the node/worker split still reads each shard exactly once per epoch. A fixed order (wds.SimpleShardList, with or
    without its seed) repeats the same shard order in every epoch.
    Full ladder (docs/ladder_plan.md section 12): with log_dir, every data worker appends "<unix time> <pass> <url>" to
    log_dir/shards_started_rank<r>.txt when it starts a shard (GPICWebDataset._log_started), and a resumed launch skips the
    shards its rank started in the pass the run stopped in, so a run that is stopped and resumed trains no sample twice
    in a pass. jobs/train_ladder.sh keeps one seed per run (shard_seed.txt), so every launch draws the same permutations,
    and keeps only the log lines up to the dump time stored in the resumed checkpoint (src/callbacks/shard_clock.py; a
    shard started later was read only for steps that the resume throws away, so it is read again). The log is read here,
    in each rank's main process before the workers start, so all workers of a rank split the same remaining list.
    Repetition arms (subsets of GPIC train, owner 2026-11-07): with workers (the training set's data workers per rank) and
    fewer shards n than B = world x workers slots, a pass holds k = B / gcd(n, B) copies of every shard (_repeat_copies),
    so every worker of every rank reads the same number S = k n / B of entries (8 shards: k 32, S 1; 80 shards: k 16, S 5)
    and the loader runs on all its workers, as on GPIC Full. The pass's schedule (_schedule) is k rounds, each a
    permutation of the n shards drawn with the seed formula above, cut into S time slots of B entries; worker w of rank r
    reads entries perm[r::world][w::workers] (in this mode this class splits by worker). All workers give one sample per
    step on average, so a slot lasts about one shard (N samples) of steps, holds each shard B / n times (32 for 8 shards,
    3 or 4 for 80), and the 8 batches of one global step (worker w of every rank: world consecutive entries) never hold
    one shard twice when n is a multiple of world. Copy i of the m copies of a shard in its g-th slot of the run starts
    at sample floor(N frac(u + g L / N + i / m)), u uniform and seeded by (seed, url), and reads L samples on from there,
    past the shard's end to its start (_RotatedTar: seeks to a sample's first tar header, no decode). L is the smallest
    sample count of the subset's shards (load_index), so every worker of every rank gives the same number of batches in a
    pass and the ranks and workers stay in step for the whole run (with whole shards they drift apart by the shards' size
    differences, and copies of one shard then meet in one step). The g L / N term starts each slot's copies where the
    slot before stopped, so the reads of an image fall about N / m steps apart all run long, also across passes: the mean
    gap of uniform random sampling (tools/sim_repeat.py). A copy leaves out N - L samples (at most 1.8% of a shard),
    and they move every slot. The log names an entry "<url>#<copy>:<k>" and the resume keys on it; a resumed pass splits the rank's remaining entries over its workers
    in schedule order (the workers' own remaining entries when the stop fell inside a slot), so no worker idles. With
    n >= B, or without workers (the validation set, tests), k = 1 and everything is as before: the same permutations,
    entries, worker split and logs."""

    def __init__(self, urls, seed, log_dir=None, workers=None):
        super().__init__()
        self.urls, self.seed, self.passes = list(urls), int(seed), 0
        self.rank, self.world = _rank_world()
        self.workers = max(1, int(workers)) if workers is not None else None
        self.k = _repeat_copies(len(self.urls), self.world * self.workers) if self.workers else 1
        self.log_path = os.path.join(log_dir, f"shards_started_rank{self.rank}.txt") if log_dir else None
        self.skip, self.index, self.length = set(), None, None
        if self.log_path:
            self.passes, self.skip = self._resume_point()
            if self.passes or self.skip:
                print(f"[shards] rank {self.rank}: resume in pass {self.passes}, {len(self.skip)} of its "
                      f"{'entries' if self.k > 1 else 'shards'} started before ({self.log_path})", flush=True)

    def _perm(self, p):
        import random
        urls = list(self.urls)
        random.Random(self.seed * 100003 + p).shuffle(urls)
        return urls

    def _schedule(self, p):
        """k > 1: the pass's entries in permutation order, each (name "<url>#<copy>:<k>", url, start fraction)."""
        import random
        rng = random.Random(self.seed * 100003 + p)
        order = []
        for _ in range(self.k):
            r = list(self.urls)
            rng.shuffle(r)
            order += r
        B = self.world * self.workers
        # a shard's g-th time slot of the run (g = p S + t, S slots per pass) starts its copies at u + g L / N + i / m: a
        # copy reads L samples on from its start, so a copy of slot g + 1 starts where one of slot g stopped, and the
        # reads of an image keep their spacing across slots and passes (with a fresh start each pass, the first reads of
        # a pass fall near the last reads of the pass before: up to 25 duplicate images in one step, tools/sim_repeat.py).
        # The N - L samples a copy leaves out move by N - L every slot. Without the index (names only), the drift is 0.
        S = self.k * len(self.urls) // B
        u0 = {u: random.Random(f"{self.seed}:{u}").random() for u in self.urls}
        drift = {u: (self.length / len(self.index[u][0]) if self.index is not None else 0.0) for u in self.urls}
        copies, out = {u: 0 for u in self.urls}, []
        for t, j in enumerate(range(0, len(order), B)):
            slot = order[j:j + B]
            m = {u: slot.count(u) for u in set(slot)}
            seen = {u: 0 for u in m}
            for u in slot:
                out.append((f"{u}#{copies[u]}:{self.k}", u, (u0[u] + (p * S + t) * drift[u] + seen[u] / m[u]) % 1.0))
                copies[u] += 1
                seen[u] += 1
        return out

    def load_index(self, threads=8):
        """k > 1: the sample offsets of every shard (tar_sample_offsets), read once in the rank's main process before the
        data workers fork, so 32 workers x 8 ranks do not each read 25K headers per entry at the start of every pass."""
        if self.k == 1:
            return
        from concurrent.futures import ThreadPoolExecutor
        import time
        t = time.time()
        with ThreadPoolExecutor(threads) as ex:
            self.index = dict(zip(self.urls, ex.map(tar_sample_offsets, self.urls)))
        self.length = min(len(st) for st, _ in self.index.values())
        print(f"[shards] rank {self.rank}: sample offsets of {len(self.urls)} shards in {time.time() - t:.1f} s; every copy "
              f"reads {self.length} samples (shards hold {self.length}-{max(len(st) for st, _ in self.index.values())})", flush=True)

    def parts(self, u, start):
        """The byte ranges of a copy of shard u that starts at fraction `start`: L samples from sample floor(start N) on,
        wrapping past the shard's end to its start."""
        starts, end = self.index[u]
        n, L = len(starts), self.length
        a = min(int(start * n), n - 1)
        b = a + L
        if b <= n:
            return [(int(starts[a]), int(starts[b]) if b < n else int(end))]
        return [(int(starts[a]), int(end)), (0, int(starts[b - n]))]

    def _names(self, p):
        """The names of this rank's entries in pass p: urls (k = 1) or "<url>#<copy>:<k>"."""
        if self.k == 1:
            return self._perm(p)[self.rank::self.world]
        return [e[0] for e in self._schedule(p)[self.rank::self.world]]

    def _resume_point(self):
        """(pass, the entries of this rank started in that pass) from the rank's log: the newest pass with a started
        entry, or the next pass when every entry of this rank in that pass was started. An entry is a shard (k = 1) or
        a copy of a shard (k > 1)."""
        try:
            with open(self.log_path) as f:
                text = f.read()
        except FileNotFoundError:
            return 0, set()
        rows = [line.split() for line in text.split("\n") if line.strip()]
        if self.k > 1 and rows and not text.endswith("\n"):   # a torn "<url>#<copy>:<k>" can still have 3 fields: no newline marks it
            rows[-1] = rows[-1][:2]
        if rows and len(rows[-1]) != 3:   # a worker killed inside its one write can leave a torn last line (review M8)
            print(f"[shards] rank {self.rank}: drops a torn last line of {self.log_path}: {' '.join(rows[-1])[:120]}", flush=True)
            rows = rows[:-1]
        if not rows:
            return 0, set()
        assert all(len(r) == 3 for r in rows), f"{self.log_path}: a line is not '<time> <pass> <url>'"
        rows = [(int(p), u) for _, p, u in rows]
        p = max(q for q, _ in rows)
        done = {u for q, u in rows if q == p}
        mine = set(self._names(p))
        assert done <= mine, (f"{self.log_path}: pass {p} names {len(done - mine)} entries that are not this rank's "
                              f"(another seed, shard list, number of GPUs or of workers?), first {sorted(done - mine)[:2]}")
        return (p + 1, set()) if done == mine else (p, done)

    def __iter__(self):
        rank, world = _rank_world()
        assert (rank, world) == (self.rank, self.world), ("rank or world size changed", (rank, world), (self.rank, self.world))
        if self.k > 1:
            yield from self._iter_copies()
            return
        perm = self._perm(self.passes)
        urls = perm[rank::world]
        n_skip = 0
        if self.skip:   # only in the first pass of a resumed launch
            urls = [u for u in urls if u not in self.skip]
            n_skip, self.skip = len(perm[rank::world]) - len(urls), set()
        wi = torch.utils.data.get_worker_info()
        if wi is None or wi.id == 0:   # one line per rank and epoch: every rank must print the same permutation
            print(f"[shards] rank {rank} pass {self.passes}: permutation {' '.join(u.rsplit('/', 1)[-1] for u in perm[:3])}; "
                  f"{len(urls)} shards for this rank" + (f", {n_skip} started before the resume skipped" if n_skip else ""),
                  flush=True)
        p = self.passes
        self.passes += 1
        for u in urls:
            yield dict(url=u, shard_pass=p)

    def _iter_copies(self):
        """k > 1: this worker's entries of the pass (the worker split is done here), without the ones the resume skips."""
        p = self.passes
        self.passes += 1
        perm = self._schedule(p)
        ents = perm[self.rank::self.world]
        wi = torch.utils.data.get_worker_info()
        if wi is not None:
            assert wi.num_workers == self.workers, ("data workers changed", wi.num_workers, self.workers)
        skip, self.skip = self.skip, set()   # only in the first pass of a resumed launch
        left = [e for e in ents if e[0] not in skip]
        if wi is None or wi.id == 0:   # one line per rank and epoch: every rank must print the same permutation
            print(f"[shards] rank {self.rank} pass {p}: {self.k} copies of {len(self.urls)} shards, permutation "
                  f"{' '.join(e[0].rsplit('/', 1)[-1] for e in perm[:3])}; {len(left)} entries for this rank, "
                  f"{len(ents) // self.workers} per worker in a whole pass"
                  + (f", {len(ents) - len(left)} started before the resume skipped" if len(left) < len(ents) else ""), flush=True)
        for name, u, start in (left if wi is None else left[wi.id::wi.num_workers]):
            d = dict(url=u, shard_pass=p, entry=name, start=start)
            if self.index is not None:
                d["parts"] = self.parts(u, start)
            yield d


def _repeat_copies(n, slots):
    """Copies of each shard per pass: 1 with n >= slots shards, else the least k with k n a multiple of slots, so every
    slot (a data worker of a rank) gets k n / slots entries: 8 shards on 8 x 32 slots -> 32 copies, 80 -> 16."""
    return 1 if n >= slots else slots // math.gcd(n, slots)


def tar_sample_offsets(path):
    """(byte offset of each sample's first tar header, end of the last member) of a shard, from the headers alone (one
    512-byte pread per member: 0.2 s for a GPIC shard of 25K members). A sample starts where the key changes, as in
    webdataset's group_by_keys. Only plain ustar members, the only kind in GPIC train; anything else raises, so a
    rotation never starts inside a sample."""
    from webdataset.tariterators import base_plus_ext
    fd = os.open(path, os.O_RDONLY)
    starts, off, prev = [], 0, None
    try:
        while True:
            h = os.pread(fd, 512, off)
            if len(h) < 512 or h == bytes(512):
                break
            if h[156:157] not in (b"0", b"\0"):
                raise ValueError(f"{path}: member at byte {off} has tar type {h[156:157]!r}; rotation reads only plain files")
            name = h[:100].split(b"\0", 1)[0].decode()
            if h[257:262] == b"ustar" and h[345:500].strip(b"\0"):
                name = h[345:500].split(b"\0", 1)[0].decode() + "/" + name
            key = base_plus_ext(name)[0]
            if key != prev:
                starts.append(off)
                prev = key
            off += 512 + (int(h[124:136].split(b"\0", 1)[0].strip() or b"0", 8) + 511) // 512 * 512
    finally:
        os.close(fd)
    assert starts, f"{path}: no members"
    return np.asarray(starts, dtype=np.int64), off


class _RotatedTar:
    """Byte ranges of a shard, each from a sample's first header to another's, then the two zero blocks that end a tar:
    a valid tar stream of those samples (EpochShuffledShards.parts, k > 1)."""

    def __init__(self, path, parts):
        self.parts = [(int(a), int(b)) for a, b in parts]
        self.f = open(path, "rb", buffering=int(os.environ.get("GOPEN_BUFFER", -1)))   # the buffer gopen gives a local file
        self.left, self.tail = 0, bytes(1024)

    def read(self, n=-1):
        out = []
        n = float("inf") if n is None or n < 0 else n
        while n > 0:
            if self.left == 0:
                if not self.parts:
                    break
                a, b = self.parts.pop(0)
                self.f.seek(a)
                self.left = b - a
                continue
            b = self.f.read(int(min(n, self.left)))
            if not b:
                raise IOError(f"{self.f.name}: short read")
            out.append(b)
            n -= len(b)
            self.left -= len(b)
        if n > 0 and self.tail:
            t, self.tail = self.tail[:int(min(n, len(self.tail)))], self.tail[int(min(n, len(self.tail))):]
            out.append(t)
        return b"".join(out)

    def close(self):
        self.f.close()


def _rotated_tar_samples(src, handler=wds.warn_and_continue):
    """wds.tarfile_to_samples for the entries of EpochShuffledShards with k > 1: each entry opened as a _RotatedTar over
    its byte ranges, then webdataset's own expander and grouping, so each sample is the same as from the plain shard."""
    from webdataset.tariterators import group_by_keys, tar_file_expander

    def opened():
        for d in src:
            # the rank's index (EpochShuffledShards.load_index) gives every entry its byte ranges; without it no entry can
            # be read, and the handler below would skip every one of them in silence
            assert "parts" in d, f"{d.get('entry')}: no byte ranges (EpochShuffledShards.load_index was not called)"
            try:
                d = dict(d, stream=_RotatedTar(d["url"], d["parts"]))
            except Exception as exn:
                exn.args = exn.args + (d.get("url"),)
                if handler(exn):
                    continue
                break
            try:
                yield d
            finally:
                d["stream"].close()

    return group_by_keys(tar_file_expander(opened(), handler=handler), handler=handler)


class _CycleBatches:
    """Profiling only: the same batches forever, so a step has no data worker, pin thread or shared memory behind it."""

    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        while True:
            yield from self.batches


def load_bad_keys(csv_path):
    bad_keys = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bad_keys.add(row["key"])
    print(f"Loaded {len(bad_keys)} corrupted keys")
    return bad_keys


def expand_pattern(pattern):
    match = re.search(r"\{(\d+)\.\.(\d+)\}", pattern)
    if not match:
        return [pattern]
    start, end = int(match.group(1)), int(match.group(2))
    width = len(match.group(1))
    return [
        pattern.replace(match.group(0), str(i).zfill(width))
        for i in range(start, end + 1)
    ]


def ignore_duplicates(exn):
    if isinstance(exn, ValueError) and "duplicate file name" in str(exn):
        return True
    return False


def make_shard_spec(tar_dir, tar_pattern):
    path = os.path.join(tar_dir, tar_pattern)

    if path.startswith("gs://"):
        return f"pipe:gcloud storage cat {path}"

    return path

# -------------------------
# GPIC WebDataset Pipeline
# -------------------------

class GPICWebDataset(IterableDataset):

    def __init__(
        self,
        tar_dir,
        tar_pattern,
        transform=None,
        shuffle_buffer=100,
        samples_per_tar=64000,
        corrupted_csv=None,
        is_val=False,
        # legacy args kept for config compatibility — not used
        caption_jsonl=None,
        output_fields=None,
        caption_types=None,
        caption_keys_pkl_file=None,
        group_text_len=False,
        group_window=1024,
        group_batch=32,
        text_max_length=320,
        pad_multiple=64,
        tokenizer_path="Qwen/Qwen3-1.7B",
        emb_dir=None,
        shared_bucket=False,
        bucket_edges=(16, 24, 32, 40, 48, 56, 64, 144, 176, 320),
        bucket_index_shards=32,
    ):
        self.tar_dir = tar_dir
        self.tar_pattern = tar_pattern
        self.is_val = is_val   # full ladder: only the training set keeps a log of started shards (EpochShuffledShards)
        self._shard_log = None
        self.shuffle_buffer = shuffle_buffer
        # ladder (option B): tokenize captions in the workers, group samples of similar caption length into the same
        # per-GPU batch, pad each batch only to its longest caption (rounded up to pad_multiple). Off by default.
        self.group_text_len = group_text_len
        self.group_window, self.group_batch = group_window, group_batch
        self.text_max_length, self.pad_multiple = text_max_length, pad_multiple
        # new ladders (owner's popup 2026-11-07 01:3x): one caption-length bucket per micro-batch for every rank
        # (_group_shared); the edges are the padded lengths, a bucket's batch pads to its edge
        self.shared_bucket = shared_bucket or (os.environ.get("GPIC_SHARED_BUCKET") == "1" and group_text_len and not is_val)
        self.bucket_edges = tuple(int(e) for e in bucket_edges)
        self.bucket_index_shards = bucket_index_shards
        assert not self.shared_bucket or (group_text_len and emb_dir), "shared_bucket needs group_text_len and the text cache"
        assert list(self.bucket_edges) == sorted(set(self.bucket_edges)) and self.bucket_edges[-1] >= text_max_length
        self.tokenizer_path = tokenizer_path
        self._tok = None
        # ladder: with emb_dir (tools/encode_captions.py) a sample carries its text encoder outputs, in place of token ids
        self.emb_dir = emb_dir
        self._emb_shard, self._emb_fd, self._emb_index = None, None, None
        self._last_sample, self._repeats = None, 0
        if emb_dir:   # a hole in the cache stops the run here: every shard of the pattern has its two files
            names = [os.path.basename(u)[:-len(".tar")] for u in expand_pattern(tar_pattern)]
            missing = [n for n in names if not (os.path.exists(os.path.join(emb_dir, n + ".idx.npz"))
                                                and os.path.exists(os.path.join(emb_dir, n + ".emb")))]
            assert not missing, f"text cache {emb_dir}: {len(missing)} of {len(names)} shards missing, first {missing[:3]}"
        self.samples_per_tar = samples_per_tar

        # print("samples_per_tar:", samples_per_tar)
        self.bad_keys = set()
        if corrupted_csv is not None:
            self.bad_keys = load_bad_keys(corrupted_csv)

        if transform is not None:
            self.transform = transform
            print(f"GPICWebDataset: using provided transform: {transform}")
        elif is_val:
            self.transform = val_transform
            print(f"GPICWebDataset: transform=None + is_val=True -> using val_transform")
        else:
            self.transform = train_transform
            print(f"GPICWebDataset: transform=None + is_val=False -> using train_transform")

        print(f"GPICWebDataset: tar_dir={tar_dir}, tar_pattern={tar_pattern}")
        print(f"  is_val   : {is_val}")
        print(f"  Bad keys : {len(self.bad_keys)}")

    # ---- filters/maps ----

    def _has_caption(self, sample):
        key = sample["__key__"]
        if key in self.bad_keys:
            return False
        meta = sample.get("json", {})
        # wds.decode auto-parses .json into a dict — if it's still bytes, parse it
        if isinstance(meta, bytes):
            try:
                meta = json.loads(meta)
            except Exception:
                return False
        return "caption" in meta and "caption_type" in meta

    def _add_caption(self, sample):
        meta = sample.get("json", {})
        if isinstance(meta, bytes):
            meta = json.loads(meta)
        sample["caption"] = meta["caption"]
        sample["caption_type"] = meta["caption_type"]
        return sample

    def __iter__(self):
        yield from self.create_dataset()

    def _apply_transform(self, sample):
        IMAGE_KEYS = ["jpg", "jpeg", "png", "webp", "JPG", "JPEG", "PNG", "WEBP"]
        for k in IMAGE_KEYS:
            if k in sample:
                img = sample[k]
                if img is None:
                    raise Exception("Decode failed — drop this sample")
                pil_img = img.convert("RGB")
                sample["image"] = self.transform(pil_img)
                break
        return sample

    # ---- ladder option B: grouped, padded-to-batch-max captions -------------------------------------------------
    def _tokenizer(self):
        if self._tok is None:   # one per worker process; the same class and call as Qwen3TextEncoder
            from transformers import Qwen2Tokenizer
            self._tok = Qwen2Tokenizer.from_pretrained(self.tokenizer_path, padding_side="right")
        return self._tok

    def _has_image(self, sample):
        return any(k in sample for k in IMAGE_SUFFIXES)

    def _crop_uint8(self, sample):
        """Decode (wds's own handler, so the same pixels as wds.decode("pil")) and the transform's first step, the random
        crop, as a uint8 array; with lever C the crop's plan (source rows, two passes) for the GPU instead."""
        for k in IMAGE_SUFFIXES:
            if k in sample:
                img = _DECODE_PIL(k, sample[k]).convert("RGB")
                if _GPU_RESIZE:
                    from .pil_exact import plan_region_crop
                    src, passes = plan_region_crop(img, 256)
                    return np.ascontiguousarray(src), passes
                return np.asarray(self._pre(img), dtype=np.uint8)
        raise ValueError("no image")

    def _tokenize(self, sample):
        ids = self._tokenizer()(sample["caption"], truncation=True, max_length=self.text_max_length)["input_ids"]
        sample["input_ids"] = ids or [self._tokenizer().eos_token_id]   # never a caption with no valid token
        sample["text_len"] = len(sample["input_ids"])
        return sample

    def _attach_emb(self, sample):
        """The sample's real-token encoder outputs, as raw bf16 bytes, from its shard's file. Read here, in tar order,
        so GPFS sees one sequential stream per worker. A failed open, a key with no entry or a short read raises through
        the pipeline and stops the run: the map has no handler, so a hole in the cache never drops samples silently."""
        shard = os.path.basename(sample["__url__"])[:-len(".tar")]
        if shard != self._emb_shard:
            if self._emb_fd is not None:
                os.close(self._emb_fd)
            # before the opens: after a failed open the next sample retries, and no closed fd number stays in use
            self._emb_shard, self._emb_fd, self._emb_index = None, None, None
            with np.load(os.path.join(self.emb_dir, shard + ".idx.npz")) as z:
                self._emb_index = dict(zip(z["keys"].tolist(), zip(z["offsets"].tolist(), z["lengths"].tolist())))
            self._emb_fd = os.open(os.path.join(self.emb_dir, shard + ".emb"), os.O_RDONLY)
            self._emb_shard = shard
        off, n = self._emb_index[sample["__key__"]]
        raw = os.pread(self._emb_fd, n * EMB_TOKEN_BYTES, off * EMB_TOKEN_BYTES)
        assert len(raw) == n * EMB_TOKEN_BYTES, (shard, sample["__key__"], len(raw), n)
        sample["text_emb"], sample["text_len"] = raw, n
        return sample

    def _group(self, src):
        """Buffer group_window samples, sort by token length, cut into chunks of group_batch, shuffle the chunk
        order, emit chunk by chunk (each chunk becomes one per-GPU batch of the DataLoader). The first window of worker i
        holds (i + 1) / num_workers of a window, so the workers read their windows from GPFS one after the other, not all
        at once every 1024 steps."""
        import random
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(hash((os.environ.get("GPIC_SHARD_SEED", "0"), wi.id if wi else 0, os.getpid()))) if not _DET else \
            random.Random(int(os.environ.get("GPIC_SHARD_SEED", "0")) * 1009 + (wi.id if wi else 0))
        buf = []

        def flush(items):
            # only full chunks: a window is a whole number of chunks, so the only partial chunk is the tail of a
            # worker's stream (< group_batch samples per worker per epoch), which is dropped. Every per-GPU batch is
            # then one full chunk, and no odd batch size reaches the compiled graphs.
            items.sort(key=lambda s: s["text_len"])
            full = [items[i:i + self.group_batch] for i in range(0, len(items) - self.group_batch + 1, self.group_batch)]
            rng.shuffle(full)
            for c in full:
                yield from c

        chunks = self.group_window // self.group_batch
        size = self.group_window if wi is None else self.group_batch * max(1, chunks * (wi.id + 1) // wi.num_workers)
        for s in src:
            buf.append(s)
            if len(buf) == size:
                yield from flush(buf)
                buf, size = [], self.group_window
        if buf:
            yield from flush(buf)

    def bucket_shares(self):
        """Each bucket's share of the samples, from the text cache's index of the first bucket_index_shards shards of the
        pattern (sorted): the same numbers in every rank and worker. Returns (edges, shares) over the non-empty buckets."""
        import bisect
        names = sorted(os.path.basename(u)[:-len(".tar")] for u in expand_pattern(self.tar_pattern))
        counts = [0] * len(self.bucket_edges)
        for n in names[:self.bucket_index_shards]:
            with np.load(os.path.join(self.emb_dir, n + ".idx.npz")) as z:
                for L in z["lengths"].tolist():
                    P = min(-(-L // self.pad_multiple) * self.pad_multiple, self.text_max_length)
                    counts[bisect.bisect_left(self.bucket_edges, P)] += 1
        keep = [i for i, c in enumerate(counts) if c]
        tot = sum(counts)
        return [self.bucket_edges[i] for i in keep], [counts[i] / tot for i in keep]

    def _group_shared(self, src):
        """New ladders (src/data/shared_bucket_loader.py): one queue per caption-length bucket (the text only: the image
        is decoded after this stage); a queue that reaches group_batch samples leaves as one batch, every sample
        tagged with its bucket's edge, which the collate pads to. A worker holds fewer than group_batch samples per
        bucket and drops nothing; SharedBucketLoader picks, on every rank alike, which bucket's batch trains next."""
        import bisect
        edges, B = self.bucket_edges, self.group_batch
        queues = [[] for _ in edges]
        for s in src:
            L = s["text_len"]
            P = min(-(-L // self.pad_multiple) * self.pad_multiple, self.text_max_length)
            q = bisect.bisect_left(edges, P)
            queues[q].append(s)
            if len(queues[q]) == B:
                for x in queues[q]:
                    x["pad_to"] = edges[q]
                    yield x
                queues[q] = []

    def _finish(self, s):
        """The image work, after grouping. A window needs only the text lengths to sort, so reading it is cheap, and a
        worker's CPU follows the batches it emits. With the images decoded in the window, every worker spends about 6 s
        of CPU at once, and the step waits for it every 1024 steps (docs/text_cache.md). Nothing after _group may drop
        a sample, or a batch would lose its size: an image that fails to decode repeats the sample before it. A fault
        on every image must not train on one image, so 8 failures in a row stop the run."""
        text = {"text_emb": s["text_emb"], "text_len": s["text_len"]} if self.emb_dir else {"input_ids": s["input_ids"]}
        if "pad_to" in s:   # shared bucket: the batch pads to its bucket's edge
            text["pad_to"] = s["pad_to"]
        try:
            # MFU lever D (docs/mfu.md): GPIC_MFU_UINT8=1 keeps the crop as uint8 HWC; the training step maps it to the same
            # float32 values the transform's ToTensor + Normalize give (uint8_to_float: a table made by those transforms)
            if _GPU_RESIZE:
                img = self._crop_uint8(s)   # (src, passes): pil_exact_gpu.pack in the collate
            else:
                img = torch.from_numpy(self._crop_uint8(s)) if _UINT8 else self._post(self._crop_uint8(s))
            out = (img, s["caption"], {"caption_type": s["caption_type"], **text})
        except Exception as exn:
            self._repeats += 1
            if self._last_sample is None or self._repeats > 8:
                raise
            print(f"[data] {s.get('__key__')}: {exn!r}: repeating the sample before it", flush=True)   # a print: patch_bugs.py filters every UserWarning
            return self._last_sample
        self._last_sample, self._repeats = out, 0
        return out

    def _collate_cached(self, samples):
        if _GPU_RESIZE:   # MFU lever C: one packed resize job for the batch
            from .pil_exact_gpu import pack
            imgs = pack([s[0] for s in samples])
        else:
            imgs = torch.stack([s[0] for s in samples])
        lens = [s[2]["text_len"] for s in samples]
        if _PACK:   # MFU lever E: the real caption tokens back to back, [sum(lens), EMB_DIM]; no mask, no padding
            emb = torch.empty((sum(lens), EMB_DIM), dtype=torch.bfloat16)
            off = 0
            for s, n in zip(samples, lens):
                emb[off:off + n] = torch.frombuffer(bytearray(s[2]["text_emb"]), dtype=torch.bfloat16).view(n, EMB_DIM)
                off += n
            return imgs, [s[1] for s in samples], {"caption_type": [s[2]["caption_type"] for s in samples],
                                                   "text_emb": emb, "text_lens": lens}
        T = samples[0][2].get("pad_to") or min(-(-max(lens) // self.pad_multiple) * self.pad_multiple, self.text_max_length)
        assert max(lens) <= T, (max(lens), T)
        emb = torch.zeros((len(samples), T, EMB_DIM), dtype=torch.bfloat16)
        text_mask = torch.zeros((len(samples), T), dtype=torch.bool)
        for r, (s, n) in enumerate(zip(samples, lens)):
            emb[r, :n] = torch.frombuffer(bytearray(s[2]["text_emb"]), dtype=torch.bfloat16).view(n, EMB_DIM)
            text_mask[r, :n] = True
        return imgs, [s[1] for s in samples], {"caption_type": [s[2]["caption_type"] for s in samples], "text_emb": emb,
                                               "text_mask": text_mask}

    def _collate_grouped(self, samples):
        imgs = torch.stack([s[0] for s in samples])
        caps = [s[1] for s in samples]
        ids = [s[2]["input_ids"] for s in samples]
        T = max(len(i) for i in ids)
        T = min(-(-T // self.pad_multiple) * self.pad_multiple, self.text_max_length)
        pad_id = self._tokenizer().pad_token_id
        input_ids = torch.full((len(ids), T), pad_id, dtype=torch.long)
        text_mask = torch.zeros((len(ids), T), dtype=torch.bool)
        for r, i in enumerate(ids):
            input_ids[r, :len(i)] = torch.tensor(i, dtype=torch.long)
            text_mask[r, :len(i)] = True
        return imgs, caps, {"caption_type": [s[2]["caption_type"] for s in samples], "input_ids": input_ids,
                            "text_mask": text_mask}

    def _log_started(self, src):
        """Full ladder, after split_by_worker: append "<unix time> <pass> <url>" to the rank's log when a worker starts a
        shard (tarfile_to_samples asks for the next url only when the tar before it is read to its end). A resume skips
        every shard started before its checkpoint, also the ones the workers were still reading: their first samples are
        in the checkpoint, so reading them again would train them twice in one pass; the rest of such a shard waits for
        the next pass (each of the rank's workers has one shard open, so a stop leaves out about half a shard per worker).
        One short O_APPEND write per line, so the workers of a rank never mix their lines."""
        import time
        fd = os.open(self._shard_log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            for d in src:
                os.write(fd, f"{time.time():.3f} {d['shard_pass']} {d.get('entry', d['url'])}\n".encode())   # entry: k > 1 only
                yield d
        finally:
            os.close(fd)

    def _shard_stages(self, shard_spec):
        """The pipeline's first stages: the shard list, the node split and the worker split. With GPIC_SHARD_SEED
        (ladder) EpochShuffledShards splits by node itself; with GPIC_SHARD_LOG_DIR as well (full ladder, training set
        only) a resumed launch skips the shards started before and every worker logs the shards it starts."""
        seed = os.environ.get("GPIC_SHARD_SEED")
        if not seed:
            return [wds.SimpleShardList(shard_spec), wds.split_by_node, wds.split_by_worker]
        import braceexpand
        urls = list(braceexpand.braceexpand(shard_spec)) if isinstance(shard_spec, str) else list(shard_spec)
        log_dir = None if self.is_val else os.environ.get("GPIC_SHARD_LOG_DIR") or None
        # repetition arms: the training set's data workers, so a subset with fewer shards than slots repeats them (k > 1)
        workers = None if self.is_val else getattr(self, "_num_workers", None)
        shards = EpochShuffledShards(urls, seed, log_dir, workers=workers)
        print(f"GPIC_SHARD_SEED={seed}: {len(urls)} shards, reshuffled every epoch"
              + (f", {shards.k} copies of each per pass (repetition arm)" if shards.k > 1 else "")
              + (f", started shards logged in {shards.log_path}" if log_dir else ""))
        self._repeat_arm = shards.k > 1
        shards.load_index()
        split = [] if self._repeat_arm else [wds.split_by_worker]   # k > 1: EpochShuffledShards splits by worker itself
        if not log_dir:
            return [shards, *split]
        self._shard_log = shards.log_path
        return [shards, *split, self._log_started]

    def create_dataset(self):
        shard_spec = make_shard_spec(self.tar_dir, self.tar_pattern)
        print("Using shard_spec:", shard_spec)

        def _has_required_keys(s):
            return (
                s is not None
                and "image" in s
                and "caption" in s
                and "caption_type" in s
            )

        # ladder: with GPIC_SHARD_SEED set (shared by all ranks through the environment; one value per run since the
        # full ladder), the shard list is shuffled before the node/worker split, with a new permutation in every
        # epoch, so no two epochs read the shards in the same order (_shard_stages). Unset: the official fixed order.
        self._repeat_arm = False
        stages = self._shard_stages(shard_spec)
        to_samples = (_rotated_tar_samples if self._repeat_arm else wds.tarfile_to_samples(handler=wds.warn_and_continue))
        if self.group_text_len:
            assert isinstance(self.transform, T.Compose) and len(self.transform.transforms) >= 2
            self._pre = self.transform.transforms[0]                        # random crop (PIL)
            self._post = T.Compose(self.transform.transforms[1:])          # ToTensor + Normalize
            return wds.DataPipeline(
                *stages,
                to_samples,                                                # wds.tarfile_to_samples unless k > 1
                wds.select(self._has_caption),
                wds.map(self._add_caption, handler=wds.warn_and_continue),
                wds.select(self._has_image),
                wds.map(self._attach_emb) if self.emb_dir else wds.map(self._tokenize, handler=wds.warn_and_continue),
                wds.shuffle(self.shuffle_buffer, **({"seed": int(os.environ.get("GPIC_SHARD_SEED", "0"))} if _DET else {})),
                self._group_shared if self.shared_bucket else self._group,   # nothing after this drops a sample
                wds.map(self._finish),                                     # the image work
            )

        dataset = wds.DataPipeline(
            *stages,

            to_samples,
            wds.decode("pil", handler=wds.warn_and_continue),

            wds.select(self._has_caption),
            wds.map(self._add_caption, handler=wds.warn_and_continue),
            wds.map(self._apply_transform, handler=wds.warn_and_continue),
            wds.select(_has_required_keys),
            wds.shuffle(self.shuffle_buffer, **({"seed": int(os.environ.get("GPIC_SHARD_SEED", "0"))} if _DET else {})),

            wds.map(
                lambda s: (
                    s["image"],
                    s["caption"],
                    {"caption_type": s["caption_type"]},
                ),
                handler=wds.warn_and_continue,
            ),
        )

        return dataset

    def build_dataloader(self, batch_size, num_workers):
        self.group_batch = batch_size   # ladder: one length-sorted chunk per per-GPU batch, whatever the config says
        self._num_workers = num_workers   # repetition arms: EpochShuffledShards sizes the copies per pass by it
        assert not self.group_text_len or self.group_window % batch_size == 0, (self.group_window, batch_size)
        dataset = self.create_dataset()
        # pinned host batches: the image copy to the GPU becomes asynchronous (Lightning moves batches with
        # non_blocking=True), so it overlaps the previous step instead of stalling the device (perf C1)
        loader = wds.WebLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
            **({"collate_fn": self._collate_cached if self.emb_dir else self._collate_grouped} if self.group_text_len else {}),
        )
        self._dataloader = loader.pipeline[0]
        if self.shared_bucket:   # new ladders: the same caption-length bucket on every rank in every micro-batch
            from src.data.shared_bucket_loader import SharedBucketLoader
            return SharedBucketLoader(loader, self.bucket_edges)
        if os.environ.get("GPIC_MFU_REBALANCE") == "2":   # MFU lever E2: the deal over the ranks on the host, one step ahead
            assert self.group_text_len and self.emb_dir and _PACK and _GPU_RESIZE, "lever E2 needs the cached, packed collate with lever C"
            from src.data.rebalance_loader import RebalancingLoader
            return RebalancingLoader(loader)
        n = int(os.environ.get("GPIC_PROF_CYCLE_BATCHES", 0))   # profiling switch (jobs/smoke_ladder_kernels.sh), unset in production
        if n and self.group_text_len:
            it = iter(loader)
            batches = [next(it) for _ in range(n)]              # real, pinned batches, with their caption lengths
            self.stop_workers()
            print(f"[cycle] {n} batches in memory, no data workers from here on", flush=True)
            return _CycleBatches(batches)
        return loader

    def stop_workers(self):
        """The data workers in the right order: the pin thread, then the workers, which exit at once. Without this call, torch's
        exit hook gives each persistent worker 5 s before it terminates it, because nobody told the worker to stop: 166 s of 8
        idle GPUs at the end of every launch with 32 workers (tools/test_worker_exit.py)."""
        it = getattr(getattr(self, "_dataloader", None), "_iterator", None)
        if it is not None:
            it._shutdown_workers()

    def test(self, dataset):
        print(f"Testing GPICWebDataset by iterating through first 5 samples...")
        for i, sample in enumerate(dataset):
            print(f"Sample {i}:")
            print(f"  Image tensor shape: {sample[0].shape}")
            print(f"  Caption: {sample[1]}")
            print(f"  Caption type: {sample[2]['caption_type']}")
            if i >= 4:
                break

    def get_length(self):
        return None


if __name__ == "__main__":
    ds = GPICWebDataset(
        tar_dir="/path_to_gpic/gpic_val_tars",
        tar_pattern="gpic_val_{00000..00031}.tar",
    )
    dataset = ds.create_dataset()
    ds.test(dataset)

_U8_TABLE = None


def uint8_to_float(x):
    """MFU lever D: uint8 images [B, H, W, 3] (any device) to the float32 [B, 3, H, W] that train_transform's ToTensor +
    Normalize give, bit for bit: a 256-entry table computed by those transforms on the CPU, looked up on x's device. A
    table avoids any question of how the device rounds x / 255 (CUDA divides by a scalar through its reciprocal)."""
    global _U8_TABLE
    if _U8_TABLE is None or _U8_TABLE.device != x.device:
        post = T.Compose(train_transform.transforms[1:])
        v = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
        t = post(v)                                   # [3, 16, 16] float32, the training transform's own numbers
        assert torch.equal(t[0], t[1]) and torch.equal(t[1], t[2])
        _U8_TABLE = t[0].reshape(256).contiguous().to(x.device)
    assert x.dtype == torch.uint8 and x.shape[-1] == 3, (x.dtype, x.shape)
    return _U8_TABLE.index_select(0, x.reshape(-1).int()).view(x.shape).permute(0, 3, 1, 2).contiguous()
