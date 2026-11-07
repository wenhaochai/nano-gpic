"""MFU lever F8 (GPIC_MFU_FP8=<recipe>, docs/mfu.md): FP8 training of the blocks' Linear layers with torchao's float8
(owner, 2026-11-06 23:5x: 「FP8肯定是要上的」). Unlike every other MFU lever this changes the numerics: the GEMMs of
the converted layers take FP8 inputs (dynamic scaling per tensor, or per row) and accumulate in fp32; a run takes it
from its first step or not at all.

Converted: every nn.Linear inside the text and image blocks (qkv / kv_y / proj of the attention, w12 / w3 of the MLP),
the GEMMs that hold nearly all of the FLOPs. Kept in bf16: adaLN_modulation (32 rows per step: nothing to gain), the
patch and caption embedders, the timestep MLP and the final layer (the first and last layers, as usual for FP8).
Recipes (torchao.float8.Float8LinearConfig.from_recipe_name): "tensorwise" (one scale per tensor, fastest),
"rowwise" (one scale per row, closer to bf16), "rowwise_with_gw_hp" (rowwise, the weight gradient in bf16).
torchao 0.11.0 (for torch 2.7) lives outside the shared env: envs/mfu-extra/torchao011 (pip --no-deps --target).
"""
import os
import sys

import torch

_PATH = os.environ.get("GPIC_TORCHAO_PATH", "/scratch/gpfs/GROUP/USER/envs/mfu-extra/torchao011")
KEEP = ("adaLN_modulation",)


def _torchao():
    if _PATH not in sys.path:
        sys.path.insert(0, _PATH)
    import torchao  # noqa: F401


def fp8_filter(mod, fqn):
    if not isinstance(mod, torch.nn.Linear):
        return False
    if not (fqn.startswith("blocks.") or fqn.startswith("text_refine_blocks.")):
        return False
    if any(k in fqn for k in KEEP):
        return False
    return mod.in_features % 16 == 0 and mod.out_features % 16 == 0


def convert_fp8(net, recipe):
    """Swap the blocks' Linear layers of net for torchao's Float8Linear (the same parameters and state-dict keys)."""
    _torchao()
    from torchao.float8 import convert_to_float8_training, Float8LinearConfig
    cfg = Float8LinearConfig.from_recipe_name(recipe)
    keys = set(net.state_dict())
    convert_to_float8_training(net, config=cfg, module_filter_fn=fp8_filter)
    assert set(net.state_dict()) == keys, "FP8 conversion changed the state-dict keys"
    n = sum(1 for m in net.modules() if type(m).__name__ == "Float8Linear")
    print(f"FP8 ({recipe}, lever F8): {n} Linear layers of the blocks converted", flush=True)
    return n
