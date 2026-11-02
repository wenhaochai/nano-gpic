import os
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

    def _crop_uint8(self, sample):
        """The transform's first step (random crop) on the decoded image, kept as a uint8 array until after
        grouping, so a 1024-sample window holds ~200 MB instead of ~800 MB of float tensors."""
        for k in ("jpg", "jpeg", "png", "webp", "JPG", "JPEG", "PNG", "WEBP"):
            if k in sample:
                img = sample[k]
                if img is None:
                    raise Exception("Decode failed — drop this sample")
                sample["image_u8"] = np.asarray(self._pre(img.convert("RGB")), dtype=np.uint8)
                break
        return sample

    def _tokenize(self, sample):
        ids = self._tokenizer()(sample["caption"], truncation=True, max_length=self.text_max_length)["input_ids"]
        sample["input_ids"] = ids
        return sample

    def _group(self, src):
        """Buffer group_window samples, sort by token length, cut into chunks of group_batch, shuffle the chunk
        order, emit chunk by chunk (each chunk becomes one per-GPU batch of the DataLoader)."""
        import random
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(hash((os.environ.get("GPIC_SHARD_SEED", "0"), wi.id if wi else 0, os.getpid())))
        buf = []

        def flush(items):
            # only full chunks: a window is a whole number of chunks, so the only partial chunk is the tail of a
            # worker's stream (< group_batch samples per worker per epoch), which is dropped. Every per-GPU batch is
            # then one full chunk, and no odd batch size reaches the compiled graphs.
            items.sort(key=lambda s: len(s["input_ids"]))
            full = [items[i:i + self.group_batch] for i in range(0, len(items) - self.group_batch + 1, self.group_batch)]
            rng.shuffle(full)
            for c in full:
                yield from c

        for s in src:
            buf.append(s)
            if len(buf) == self.group_window:
                yield from flush(buf)
                buf = []
        if buf:
            yield from flush(buf)

    def _finish(self, s):
        return (self._post(s["image_u8"]), s["caption"], {"caption_type": s["caption_type"], "input_ids": s["input_ids"]})

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
                wds.decode("pil", handler=wds.warn_and_continue),
                wds.select(self._has_caption),
                wds.map(self._add_caption, handler=wds.warn_and_continue),
                wds.map(self._crop_uint8, handler=wds.warn_and_continue),
                wds.select(lambda s: s is not None and "image_u8" in s),
                wds.map(self._tokenize, handler=wds.warn_and_continue),
                wds.shuffle(self.shuffle_buffer),
                self._group,                                               # nothing after this drops a sample
                wds.map(self._finish),
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
            **({"collate_fn": self._collate_grouped} if self.group_text_len else {}),
        )
        return loader

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