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

from .augmentation import center_crop_arr, random_crop_arr
global train_transform, val_transform

train_transform = T.Compose([
    T.Lambda(lambda pil_image: random_crop_arr(pil_image, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])
print("Defined train_transform function (from dataset.augmentation import random_crop_arr) :", train_transform)

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

class EpochShuffledShards(IterableDataset):
    """ladder: the shard list, reshuffled at every pass with Random(seed * 100003 + pass). Every rank and worker holds
    its own copy and makes exactly one pass per epoch (persistent workers call iter() once per epoch), so in epoch e
    every copy uses the same permutation and the node/worker split still reads each shard exactly once per epoch.
    A fixed order (wds.SimpleShardList, with or without its seed) repeats the same shard order in every epoch."""

    def __init__(self, urls, seed):
        super().__init__()
        self.urls, self.seed, self.passes = list(urls), int(seed), 0

    def __iter__(self):
        import random
        urls = list(self.urls)
        random.Random(self.seed * 100003 + self.passes).shuffle(urls)
        wi = torch.utils.data.get_worker_info()
        if wi is None or wi.id == 0:   # one line per rank and epoch: every rank must print the same shards
            dist = torch.distributed
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            print(f"[shards] rank {rank} pass {self.passes}: {' '.join(u.rsplit('/', 1)[-1] for u in urls[:3])}",
                  flush=True)
        self.passes += 1
        for u in urls:
            yield dict(url=u)


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
    ):
        self.tar_dir = tar_dir
        self.tar_pattern = tar_pattern
        self.shuffle_buffer = shuffle_buffer
        # ladder (option B): tokenize captions in the workers, group samples of similar caption length into the same
        # per-GPU batch, pad each batch only to its longest caption (rounded up to pad_multiple). Off by default.
        self.group_text_len = group_text_len
        self.group_window, self.group_batch = group_window, group_batch
        self.text_max_length, self.pad_multiple = text_max_length, pad_multiple
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
        crop, as a uint8 array."""
        for k in IMAGE_SUFFIXES:
            if k in sample:
                return np.asarray(self._pre(_DECODE_PIL(k, sample[k]).convert("RGB")), dtype=np.uint8)
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
        rng = random.Random(hash((os.environ.get("GPIC_SHARD_SEED", "0"), wi.id if wi else 0, os.getpid())))
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

    def _finish(self, s):
        """The image work, after grouping. A window needs only the text lengths to sort, so reading it is cheap, and a
        worker's CPU follows the batches it emits. With the images decoded in the window, every worker spends about 6 s
        of CPU at once, and the step waits for it every 1024 steps (docs/text_cache.md). Nothing after _group may drop
        a sample, or a batch would lose its size: an image that fails to decode repeats the sample before it. A fault
        on every image must not train on one image, so 8 failures in a row stop the run."""
        text = {"text_emb": s["text_emb"], "text_len": s["text_len"]} if self.emb_dir else {"input_ids": s["input_ids"]}
        try:
            out = (self._post(self._crop_uint8(s)), s["caption"], {"caption_type": s["caption_type"], **text})
        except Exception as exn:
            self._repeats += 1
            if self._last_sample is None or self._repeats > 8:
                raise
            warnings.warn(f"{s.get('__key__')}: {exn!r}: repeating the sample before it")
            return self._last_sample
        self._last_sample, self._repeats = out, 0
        return out

    def _collate_cached(self, samples):
        imgs = torch.stack([s[0] for s in samples])
        lens = [s[2]["text_len"] for s in samples]
        T = min(-(-max(lens) // self.pad_multiple) * self.pad_multiple, self.text_max_length)
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

        # ladder: with GPIC_SHARD_SEED set (one value per launch, shared by all ranks through the environment), the
        # shard list is shuffled before the node/worker split, with a new permutation in every epoch, so a resumed
        # run does not re-read the same leading shards of every worker and no two epochs read the shards in the same
        # order. Unset: the official fixed order.
        shards = wds.SimpleShardList(shard_spec)
        seed = os.environ.get("GPIC_SHARD_SEED")
        if seed:
            import braceexpand
            urls = list(braceexpand.braceexpand(shard_spec)) if isinstance(shard_spec, str) else list(shard_spec)
            shards = EpochShuffledShards(urls, seed)
            print(f"GPIC_SHARD_SEED={seed}: {len(urls)} shards, reshuffled every epoch")
        if self.group_text_len:
            assert isinstance(self.transform, T.Compose) and len(self.transform.transforms) >= 2
            self._pre = self.transform.transforms[0]                        # random crop (PIL)
            self._post = T.Compose(self.transform.transforms[1:])          # ToTensor + Normalize
            return wds.DataPipeline(
                shards,
                wds.split_by_node,
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=wds.warn_and_continue),
                wds.select(self._has_caption),
                wds.map(self._add_caption, handler=wds.warn_and_continue),
                wds.select(self._has_image),
                wds.map(self._attach_emb) if self.emb_dir else wds.map(self._tokenize, handler=wds.warn_and_continue),
                wds.shuffle(self.shuffle_buffer),
                self._group,                                               # nothing after this drops a sample
                wds.map(self._finish),                                     # the image work
            )

        dataset = wds.DataPipeline(
            shards,
            wds.split_by_node,
            wds.split_by_worker,

            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.decode("pil", handler=wds.warn_and_continue),

            wds.select(self._has_caption),
            wds.map(self._add_caption, handler=wds.warn_and_continue),
            wds.map(self._apply_transform, handler=wds.warn_and_continue),
            wds.select(_has_required_keys),
            wds.shuffle(self.shuffle_buffer),

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