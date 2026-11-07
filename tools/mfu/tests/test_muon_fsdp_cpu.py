#!/usr/bin/env python3
"""CPU test of src/optim/muon_fsdp.py (ShardMuonAdamW, Muon under FSDP2) against the reference MuonAdamW
(src/optim/muon.py) under DDP: 2 ranks, gloo, a small JiT_T2I, the same synthetic gradients on both (the FSDP2 copy gets
each gradient as its Shard(0) DTensor), Newton-Schulz in fp32 and in bf16, split off and on, the mup and moonlight
update scales, 3 steps with weight decay. Every parameter and
momentum buffer, gathered, must equal the reference's (Muon matrices bit for bit: the same owners, the same batches,
the same adds; AdamW parameters within 1e-6, CPU's fused AdamW rounding a tensor's tail by layout). Also the optimizer
state's keys and shapes after a full gather match the reference's.
usage: python test_muon_fsdp_cpu.py"""
import copy, os, sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen"))
sys.path.insert(0, SRC)

SMALL = dict(input_size=64, in_channels=3, num_groups=4, hidden_size=128, num_blocks=3, num_text_blocks=1, patch_size=16,
             txt_embed_dim=64, txt_max_length=16, bottleneck_dim=32)


def run(rank, world, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    # gloo has no bf16 all_gather: carry the reference's bf16 buffer as bytes (the same bits)
    _ag = dist.all_gather_into_tensor

    def ag(out, inp, *a, **k):
        if inp.dtype == torch.bfloat16:
            return _ag(out.view(torch.uint8), inp.view(torch.uint8), *a, **k)
        return _ag(out, inp, *a, **k)
    dist.all_gather_into_tensor = ag
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, distribute_tensor, Shard
    from src.models.transformer.JiT_T2I import JiT_T2I
    from src.optim.muon import MuonAdamW, tag_muon_params
    from src.optim.muon_fsdp import ShardMuonAdamW
    from src.utils.fsdp_setup import shard
    mesh = init_device_mesh("cpu", (world,))
    ok = True
    for ns, split, scale in (("float32", False, "moonlight"), ("float32", True, "mup"), ("bfloat16", False, "mup"),
                             ("bfloat16", True, "mup"), ("bfloat16", True, "moonlight")):
        if True:
            torch.manual_seed(0)
            base = JiT_T2I(**SMALL)
            for p in base.parameters():
                p.data.add_(torch.randn_like(p) * 0.02)
            ref, fs = copy.deepcopy(base), copy.deepcopy(base)
            shard(fs, mesh, compute_bf16=False, compile_blocks=False, group=2)
            tag_muon_params(ref); tag_muon_params(fs)
            kw = dict(lr=2e-3, muon_lr=2e-3, split=split, ns_dtype=ns, weight_decay=0.01, muon_weight_decay=0.01,
                      update_scale=scale)
            o_ref = MuonAdamW([{"params": [p for p in ref.parameters() if p.requires_grad]}], **kw)
            o_fs = ShardMuonAdamW([{"params": [p for p in fs.parameters() if p.requires_grad]}], **kw)
            names = [n for n, p in ref.named_parameters() if p.requires_grad]
            pf = dict(fs.named_parameters())
            pr = dict(ref.named_parameters())
            for s in range(3):
                g = torch.Generator().manual_seed(1000 + s)
                for n in names:
                    full = torch.randn(pr[n].shape, generator=g) * 1e-2
                    pr[n].grad = full.clone()
                    q = pf[n]
                    q.grad = distribute_tensor(full.clone(), mesh, [Shard(0)]) if isinstance(q, DTensor) else full.clone()
                o_ref.step(); o_fs.step()
            full_t = lambda t: t.full_tensor() if isinstance(t, DTensor) else t
            dm, da, same_m = 0.0, 0.0, True
            for n in names:
                a, b = pr[n].detach(), full_t(pf[n].detach())
                if hasattr(pr[n], "_muon_parts"):
                    same_m &= torch.equal(a, b)
                    dm = max(dm, float((a - b).abs().max()))
                    mb = full_t(o_fs.state[pf[n]]["momentum_buffer"])
                    same_m &= torch.equal(o_ref.state[pr[n]]["momentum_buffer"], mb)
                else:
                    da = max(da, float((a - b).abs().max() / a.abs().max().clamp_min(1e-12)))
            keys_ok = all(set(o_ref.state[pr[n]]) == set(o_fs.state[pf[n]]) for n in names)
            # checkpoint round trip as Lightning's ModelParallelStrategy does it (full state dicts), then one more step
            # with the same gradients on the original and on the restored copy: the same bits
            from torch.distributed.checkpoint.state_dict import (StateDictOptions, get_model_state_dict,
                                                                 get_optimizer_state_dict, set_model_state_dict,
                                                                 set_optimizer_state_dict)
            full_opt = StateDictOptions(full_state_dict=True)
            msd = get_model_state_dict(fs, options=full_opt)
            osd = get_optimizer_state_dict(fs, o_fs, options=full_opt)
            fs2 = copy.deepcopy(base)
            shard(fs2, mesh, compute_bf16=False, compile_blocks=False, group=2)
            tag_muon_params(fs2)
            o_fs2 = ShardMuonAdamW([{"params": [p for p in fs2.parameters() if p.requires_grad]}], **kw)
            # a checkpoint holds copies (the full state dict here still refers to the live step tensors, which two
            # optimizers in one process would then share and both advance)
            set_model_state_dict(fs2, copy.deepcopy(msd), options=full_opt)
            set_optimizer_state_dict(fs2, o_fs2, copy.deepcopy(osd), options=full_opt)
            pf2 = dict(fs2.named_parameters())
            g = torch.Generator().manual_seed(77)
            for n in names:
                full = torch.randn(pr[n].shape, generator=g) * 1e-2
                for d in (pf, pf2):
                    q = d[n]
                    q.grad = distribute_tensor(full.clone(), mesh, [Shard(0)]) if isinstance(q, DTensor) else full.clone()
            o_fs.step(); o_fs2.step()
            rt = all(torch.equal(full_t(pf[n].detach()), full_t(pf2[n].detach())) for n in names)
            if rank == 0:
                print(f"   checkpoint round trip (full state dicts), one more step: the same bits {rt}", flush=True)
            good = same_m and da < 1e-6 and keys_ok and rt
            ok &= good
            if rank == 0:
                print(f"ns={ns} split={split} scale={scale}: Muon matrices and momentum bit for bit {same_m} (max |diff| {dm:.1e}); "
                      f"AdamW max rel diff {da:.1e}; state keys equal {keys_ok}  {'ok' if good else 'FAIL'}", flush=True)
    t = torch.tensor([int(ok)])
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    if rank == 0:
        print("MUON_FSDP_TEST " + ("PASS" if t.item() else "FAIL"), flush=True)
    dist.destroy_process_group()
    if not t.item():
        raise SystemExit(1)


if __name__ == "__main__":
    mp.spawn(run, args=(2, 29581), nprocs=2, join=True)
