"""ladder: the text encoder's outputs read from the batch, not computed in the training step.

The frozen encoder inside the step makes every step wait for the rank with the longest captions (docs/text_cache.md):
T averages 51 per GPU, the longest of 8 GPUs averages 131, so the encoder costs about 26 ms per step at any model size.
tools/encode_captions.py computes the outputs once, with the training path's tokenizer, padding, kernel and bf16
autocast, and the data path attaches them (GPICWebDataset(emb_dir=...)). Real tokens carry the same values as the
online encoder up to bf16 rounding; padded tokens are zeros, and the denoiser's masks keep them out of every attention.
Prediction keeps the online encoder (configs_h100/lite_ladder_sample.yaml).
"""
import os

import torch

from src.models.conditioner.base import BaseConditioner


def assert_left_aligned(mask):
    """MFU review M3: FA3 (lever A) takes each sample's first mask.sum() caption keys as its valid ones, so a mask with
    padding before or between tokens would let it attend to padding. A device-side assert (eager, no host sync)."""
    n = mask.sum(1, keepdim=True)
    torch._assert_async((mask == (torch.arange(mask.shape[1], device=mask.device)[None] < n)).all())


class CachedTextConditioner(BaseConditioner):
    def __init__(self, emb_dir: str, embed_dim: int = 2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.null_cpu = torch.load(os.path.join(emb_dir, "null.pt"), map_location="cpu", weights_only=True)
        assert self.null_cpu.dtype == torch.bfloat16 and self.null_cpu.shape == (1, 1, embed_dim), self.null_cpu.shape
        self.null_embedding = None

    @staticmethod
    def _to_cuda(t):
        return t if t.is_cuda else t.pin_memory().cuda(non_blocking=True)

    def _impl_condition(self, y, metadata: dict = {}):
        if "text_lens" in metadata:   # MFU lever E: packed captions [S, embed_dim]; y_pack comes from src/utils/rebalance.py
            return self._to_cuda(metadata["text_emb"])
        out, mask = self._to_cuda(metadata["text_emb"]), self._to_cuda(metadata["text_mask"])
        metadata["y_mask"] = mask.bool()
        assert_left_aligned(metadata["y_mask"])
        return out

    def _impl_uncondition(self, y, metadata: dict = None):
        if "text_lens" in metadata:   # MFU lever E: no caption dropout in packed mode (BaseTrainer.preproprocess refuses it)
            return metadata["text_emb"][:0]
        # the null caption: one eos token at position 0, zeros after; the CFG-dropout code sets its mask row
        if self.null_embedding is None:
            self.null_embedding = self.null_cpu.to(metadata["text_emb"].device)
        T = metadata["y_mask"].shape[1]
        out = torch.zeros(len(y), T, self.embed_dim, device=self.null_embedding.device, dtype=self.null_embedding.dtype)
        out[:, 0] = self.null_embedding[0, 0]
        return out
