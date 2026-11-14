#!/usr/bin/env python3
"""GPU test of GPIC_MUON_FUSE=1 (src/optim/muon_add.py nesterov_units_triton and decay_add_units_triton; src/optim/muon_fsdp.py).

1. The kernels against ATen's passes, bit for bit: lerp_ (momentum), lerp (Nesterov) and the cast to bf16; mul_ (weight decay) and the
   add; several tensors per launch, sizes below, at and above the kernel's block, an empty tensor, two value scales, with and without
   Nesterov, with and without weight decay.
2. ShardMuonAdamW under FSDP2 on 2 GPUs (NCCL), a small JiT_T2I (MUON_TEST_MODEL as test_muon_fsdp_cpu.py: e.g. an MoE model), bf16
   Newton-Schulz: two copies, one stepped with the fuse off and one with it on, the same synthetic gradients, 4 steps with weight decay:
   every parameter and momentum buffer the same bits. Run it also with GPIC_MUON_LOCAL=1 (the experts' units owned where they live).
usage: python test_muon_fuse.py   (2 CUDA GPUs for part 2; with one GPU only part 1 runs)"""
import copy, json, os, sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen"))
sys.path.insert(0, SRC)
SMALL = dict(input_size=64, in_channels=3, num_groups=4, hidden_size=128, num_blocks=3, num_text_blocks=1, patch_size=16,
             txt_embed_dim=64, txt_max_length=16, bottleneck_dim=32)
SMALL.update(json.loads(os.environ.get("MUON_TEST_MODEL", "{}")))


def kernels():
    from src.optim.muon_add import add_units_triton, decay_add_units_triton, nesterov_units_triton
    dev, ok = "cuda", True
    torch.manual_seed(0)
    sizes = [1, 4095, 4096, 4097, 0, 100000, 1 << 22]
    for scale in (1.0, 1e-3):
        for nest in (True, False):
            beta = 0.95
            g = [torch.randn(n, device=dev) * scale for n in sizes]
            m = [torch.randn(n, device=dev) * scale * 0.3 for n in sizes]
            m_ref = [x.clone() for x in m]
            torch._foreach_lerp_(m_ref, g, 1 - beta)
            u_ref = [(a.lerp(b, beta) if nest else b).to(torch.bfloat16) for a, b in zip(g, m_ref)]
            u = [torch.empty(n, dtype=torch.bfloat16, device=dev) for n in sizes]
            nesterov_units_triton(g, m, u, beta, nest)
            dm = sum(int((a != b).sum()) for a, b in zip(m, m_ref)); du = sum(int((a != b).sum()) for a, b in zip(u, u_ref))
            good = dm == 0 and du == 0
            ok &= good
            print(f"{'ok  ' if good else 'FAIL'} momentum and Nesterov input, scale {scale}, nesterov {nest}: {dm} momentum values and {du} bf16 "
                  f"inputs of {sum(sizes)} differ from ATen's", flush=True)
        for wd in (0.01807, 0.0):
            lr = 9.617e-3
            sz = [n for n in sizes if n]
            w = [torch.randn(n, device=dev) * scale for n in sz]
            o = [torch.randn(n, device=dev).to(torch.bfloat16) for n in sz]
            al = [-lr * c for c in (0.71, 1.0, 0.33, 1.41, 2.0, 0.5)][:len(sz)]
            w_ref = [x.clone() for x in w]
            if wd:
                torch._foreach_mul_(w_ref, 1 - lr * wd)
            add_units_triton(w_ref, o, al)
            w_ref2 = [x.clone() for x in w]
            if wd:
                torch._foreach_mul_(w_ref2, 1 - lr * wd)
            for x, y, a in zip(w_ref2, o, al):
                x.add_(y, alpha=a)
            decay_add_units_triton(w, o, al, 1 - lr * wd if wd else 1.0)
            d1 = sum(int((a != b).sum()) for a, b in zip(w, w_ref)); d2 = sum(int((a != b).sum()) for a, b in zip(w, w_ref2))
            good = d1 == 0 and d2 == 0
            ok &= good
            print(f"{'ok  ' if good else 'FAIL'} decay and add, scale {scale}, weight decay {wd}: {d1} of {sum(sz)} values differ from mul_ + the "
                  f"Triton add, {d2} from mul_ + add_", flush=True)
    return ok


def run(rank, world, port, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, distribute_tensor, Shard
    from src.models.transformer.JiT_T2I import JiT_T2I
    from src.optim.muon import tag_muon_params
    import src.optim.muon_fsdp as mf
    from src.utils.fsdp_setup import shard
    mesh = init_device_mesh("cuda", (world,))
    dev = torch.device("cuda", rank)
    ok = True
    for split, scale, nest, wd in ((True, "mup", True, 0.01), (False, "moonlight", True, 0.01), (True, "mup", False, 0.0)):
        torch.manual_seed(0)
        base = JiT_T2I(**SMALL)
        for p in base.parameters():
            p.data.add_(torch.randn_like(p) * 0.02)
        nets, opts = [], []
        for _ in range(2):
            net = copy.deepcopy(base).to(dev)
            shard(net, mesh, compute_bf16=False, compile_blocks=False, group=2)
            tag_muon_params(net)
            nets.append(net)
            opts.append(mf.ShardMuonAdamW([{"params": [p for p in net.parameters() if p.requires_grad]}], lr=2e-3, muon_lr=2e-3, split=split,
                                          ns_dtype="bfloat16", weight_decay=wd, muon_weight_decay=wd, update_scale=scale))
        for o in opts:
            for gr in o.param_groups:
                if gr.get("use_muon"):
                    gr["nesterov"] = nest
        names = [n for n, p in nets[0].named_parameters() if p.requires_grad]
        ps = [dict(net.named_parameters()) for net in nets]
        for s in range(4):
            g = torch.Generator().manual_seed(1000 + s)
            for n in names:
                full = (torch.randn(ps[0][n].shape, generator=g) * 1e-2).to(dev)
                for d in ps:
                    qp = d[n]
                    qp.grad = distribute_tensor(full.clone(), mesh, [Shard(0)]) if isinstance(qp, DTensor) else full.clone()
            mf._MUON_FUSE = False; opts[0].step()
            mf._MUON_FUSE = True; opts[1].step()
        mf._MUON_FUSE = False
        loc = lambda t: t.to_local() if isinstance(t, DTensor) else t
        same, nm, cov = True, 0, opts[1]._plan.get("covered")
        for n in names:
            same &= torch.equal(loc(ps[0][n].detach()), loc(ps[1][n].detach()))
            st0, st1 = opts[0].state[ps[0][n]], opts[1].state[ps[1][n]]
            if "momentum_buffer" in st0:
                nm += 1
                same &= torch.equal(loc(st0["momentum_buffer"]), loc(st1["momentum_buffer"]))
        t = torch.tensor([int(same and bool(cov) and nm > 0)], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        ok &= bool(t.item())
        if rank == 0:
            print(f"{'ok  ' if t.item() else 'FAIL'} FSDP2 on {world} GPUs, split {split}, scale {scale}, nesterov {nest}, weight decay {wd}, "
                  f"GPIC_MUON_LOCAL={os.environ.get('GPIC_MUON_LOCAL', '0')}: fuse on = fuse off in every parameter and momentum buffer "
                  f"({nm} Muon tensors; the decay inside the add: {cov})", flush=True)
    q.put(ok)
    dist.destroy_process_group()


if __name__ == "__main__":
    OK = kernels()
    if torch.cuda.device_count() >= 2:
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        procs = [ctx.Process(target=run, args=(r, 2, 29593, q)) for r in range(2)]
        for p in procs:
            p.start()
        res = [q.get(timeout=900) for _ in procs]
        for p in procs:
            p.join()
        OK &= all(res) and all(p.exitcode == 0 for p in procs)
    else:
        print("one GPU: the FSDP2 part did not run")
    print("MUON_FUSE_TEST " + ("PASS" if OK else "FAIL")); sys.exit(0 if OK else 1)
