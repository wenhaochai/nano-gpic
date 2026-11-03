import os

import torch
import torch.nn as nn
from src.models.conditioner.base import BaseConditioner

from transformers import Qwen3Model, Qwen2Tokenizer


class Qwen3TextEncoder(BaseConditioner):
    def __init__(self, weight_path: str, embed_dim:int=None, max_length=128, text_mode: str = "official",
                 attn_implementation: str = "flex_attention", pad_multiple: int = 64):
        super().__init__()
        # ladder (option B): text_mode="masked" pads each batch only to its longest caption (rounded up to
        # pad_multiple, never truncating below max_length) and hands the padding mask to the denoiser; the null
        # caption for CFG dropout is one eos token. "official" keeps the original behaviour.
        assert text_mode in ("official", "masked"), text_mode
        self.text_mode, self.pad_multiple = text_mode, pad_multiple
        self.null_embedding = None
        self.tokenizer = Qwen2Tokenizer.from_pretrained(weight_path, max_length=max_length, padding_side="right")
        # perf C6: FlexAttention for the causal + padding mask (same masking; SDPA had to fall back to the
        # memory-efficient kernel for this mask, 24% tensor-core active under ncu)
        self.model = Qwen3Model.from_pretrained(weight_path, attn_implementation=attn_implementation).to(torch.bfloat16)
        # profiling switches (jobs/prof_ladder.sh), unset in every production run
        self.model.compile(**({"mode": os.environ["GPIC_PROF_TE_COMPILE_MODE"]} if "GPIC_PROF_TE_COMPILE_MODE" in os.environ else {}))
        self.cudagraphs = os.environ.get("GPIC_PROF_TE_COMPILE_MODE") == "reduce-overhead"
        self.no_padding_mask = bool(os.environ.get("GPIC_PROF_TE_NO_MASK"))
        self.extra = {"use_cache": False} if os.environ.get("GPIC_PROF_TE_NO_CACHE") else {}
        self.uncondition_embedding = None
        self.embed_dim = embed_dim
        self.max_length = max_length
        # torch._dynamo.config.optimize_ddp = False

    @staticmethod
    def _to_cuda(t):
        return t if t.is_cuda else t.pin_memory().cuda(non_blocking=True)

    def _impl_condition(self, y, metadata:dict={}):
        if self.text_mode == "masked":
            if isinstance(metadata, dict) and metadata.get("input_ids") is not None:   # tokenized in the workers
                input_ids, mask = self._to_cuda(metadata["input_ids"]), self._to_cuda(metadata["text_mask"])
            else:   # strings (prediction): pad to the longest caption, rounded up to pad_multiple
                tok = self.tokenizer(y, truncation=True, max_length=self.max_length, padding="longest",
                                     pad_to_multiple_of=self.pad_multiple, return_tensors="pt")
                input_ids, mask = self._to_cuda(tok.input_ids), self._to_cuda(tok.attention_mask.bool())
            # with right padding the causal mask already keeps a real token from seeing padding (no_padding_mask)
            if self.cudagraphs:
                torch.compiler.cudagraph_mark_step_begin()
            out = self.model(input_ids=input_ids, attention_mask=None if self.no_padding_mask else mask.long(), **self.extra)[0]
            if self.cudagraphs:   # a CUDA graph's outputs live in its static memory, overwritten by the next call
                out = out.clone()
            if isinstance(metadata, dict):
                metadata["y_mask"] = mask.bool()
            return self._fit_dim(out)
        tokenized = self.tokenizer(y, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        # perf C7: pinned, asynchronous copies (a pageable copy blocks the host until the GPU queue drains)
        input_ids = tokenized.input_ids.pin_memory().cuda(non_blocking=True)
        attention_mask = tokenized.attention_mask.pin_memory().cuda(non_blocking=True)
        metadata["valid_length_y"] = torch.sum(attention_mask, dim=-1)
        y = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        if y.shape[2] < self.embed_dim:
            y = torch.cat([y, torch.zeros(y.shape[0], y.shape[1], self.embed_dim - y.shape[2]).to(y.device, y.dtype)], dim=-1)
        if y.shape[2] > self.embed_dim:
            y = y[:, :, :self.embed_dim]
        return y

    def _fit_dim(self, y):
        if y.shape[2] < self.embed_dim:
            y = torch.cat([y, torch.zeros(y.shape[0], y.shape[1], self.embed_dim - y.shape[2]).to(y.device, y.dtype)], dim=-1)
        if y.shape[2] > self.embed_dim:
            y = y[:, :, :self.embed_dim]
        return y

    def _impl_uncondition(self, y, metadata:dict=None):
        if self.text_mode == "masked":
            # null caption: one eos token at position 0, zeros after; the CFG-dropout code sets its mask row
            if self.null_embedding is None:
                ids = torch.tensor([[self.tokenizer.eos_token_id]], device=self.model.device)
                if self.cudagraphs:
                    torch.compiler.cudagraph_mark_step_begin()
                self.null_embedding = self._fit_dim(self.model(input_ids=ids, attention_mask=torch.ones_like(ids), **self.extra)[0]).clone()
            T = metadata["y_mask"].shape[1] if isinstance(metadata, dict) and "y_mask" in metadata else 1
            out = torch.zeros(len(y), T, self.null_embedding.shape[2], device=self.null_embedding.device,
                              dtype=self.null_embedding.dtype)
            out[:, 0] = self.null_embedding[0, 0]
            return out
        if self.uncondition_embedding is not None and "negative_prompt" not in metadata:
            return self.uncondition_embedding.repeat(len(y), 1, 1)
        negative_prompt = "" if "negative_prompt" not in metadata else metadata['negative_prompt']
        self.uncondition_embedding = self._impl_condition([negative_prompt,])
        return self.uncondition_embedding.repeat(len(y), 1, 1)