"""
Dataset for PixelGen t2i predict mode driven by a JSONL prompt file.

Each line in the JSONL must have:
  - "key"          : unique image identifier (str)
  - "caption"      : text prompt (str)
  - "caption_type" : e.g. "short", "medium", "long", "tag"  (str)

Saved filename: {caption_type}_{key}.png  — mirrors the LlamaGen GPIC convention.
"""

import torch
import json
import os
from torch.utils.data import Dataset
from PIL import Image


def gpic_jsonl_save_fn(image, metadata, root_path):
    """Save a single HWC uint8 numpy array as PNG."""
    fname = f"{metadata['caption_type']}_{metadata['key']}.png"
    Image.fromarray(image).save(os.path.join(root_path, fname))


class GPICJsonlDataset(Dataset):
    """
    Reads prompts from a JSONL file and produces (latent, condition, metadata)
    tuples compatible with PixelGen's predict pipeline.

    Args:
        jsonl_path:             Path to the JSONL file.
        num_samples_per_instance: Number of independent noise samples per prompt.
        latent_shape:           Shape of the initial noise tensor, e.g. [3, 256, 256].
    """

    def __init__(self, jsonl_path: str, num_samples_per_instance: int, latent_shape: list):
        self.jsonl_path = jsonl_path
        self.latent_shape = latent_shape
        self.num_samples_per_instance = num_samples_per_instance

        with open(jsonl_path) as f:
            self.records = [json.loads(line) for line in f if line.strip()]

        self.num_instances = len(self.records)
        self.num_samples = self.num_instances * num_samples_per_instance

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        instance_idx = idx // self.num_samples_per_instance
        sample_idx   = idx % self.num_samples_per_instance

        record = self.records[instance_idx]

        # Reproducible noise — same seed as GenEvalDataset convention
        generator = torch.Generator().manual_seed(sample_idx)
        latent    = torch.randn(self.latent_shape, generator=generator, dtype=torch.float32)

        condition = record["caption"]

        metadata = {
            "key":          record["key"],
            "caption":      record["caption"],
            "caption_type": record["caption_type"],
            "seed":         sample_idx,
            "save_fn":      gpic_jsonl_save_fn,
        }

        return latent, condition, metadata