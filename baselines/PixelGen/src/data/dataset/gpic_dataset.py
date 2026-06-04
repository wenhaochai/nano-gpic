import os
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
    ):
        self.tar_dir = tar_dir
        self.tar_pattern = tar_pattern
        self.shuffle_buffer = shuffle_buffer
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

        dataset = wds.DataPipeline(
            wds.SimpleShardList(shard_spec),
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
        dataset = self.create_dataset()
        loader = wds.WebLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
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