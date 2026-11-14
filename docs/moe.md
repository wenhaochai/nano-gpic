# MoE line: infra notes (branch moe)

Plan: `tmp/ladder_prof/moe/PLAN.md` (owner approval 2026-11-13 22:18). Ledger: `project/gpic/logs/sweep_moe.yaml`.
Layer: `src/models/layers/moe.py`; balance and monitoring: `src/callbacks/moe_balance.py`; Triton sums (off by default):
`src/models/layers/moe_ops.py`.

## Step time and memory on 8 H100 (infra smoke w2680, 2026-11-14)

d12 = 1024 x (12+3), 128 images per GPU, 400 steps per arm, frozen worktree `gpic-wt/moe_smoke1` (moe b3ead47).
GPU time per step: median after step 20, first to last kernel of a step. X = expert parameters / dense FFN parameters,
G = active experts per token.

| arm | spec | mode | parameters | s/step | x dense | peak GiB |
|---|---|---|---|---|---|---|
| dense | dense | DDP | 0.38B | 0.1218 | 1.00 | 43.0 |
| ecb8 | ecb:8:2 (X 4) | DDP | 0.83B | 0.1633 | 1.34 | 57.4 |
| ecb16 | ecb:16:2 (X 8) | DDP | 1.43B | 0.1899 | 1.56 | 68.5 |
| eci16 | eci:16:2 | DDP | 1.43B | 0.1886 | 1.55 | 68.7 |
| tc16 | tc:16:2, capacity 1.5 | DDP | 1.43B | 0.2163 | 1.78 | 73.3 |
| ecb64g8 | ecb:64:8 (X 8, G 8) | DDP | 1.43B | 0.2203 | 1.81 | 77.9 |
| ecb16fs | ecb:16:2 | FSDP2 | 1.43B | 0.1623 | 1.33 | 49.7 |
| ecb32fs | ecb:32:2 (X 16) | FSDP2 | 2.49B | 0.1925 | 1.58 | 55.0 |

- Most of the MoE overhead scales with the parameter count, not with the routing. Under DDP: 44 ms per step per 1e9
  parameters (dense, ecb8, ecb16) plus about 22 ms for the routed layers. Under FSDP2: 28.5 ms per 1e9 (ecb16fs and
  ecb32fs have the same compute per token and differ only in expert parameters).
- FSDP2 is faster for every MoE shape here, needs about 19 GiB less, and compiles in 70 s instead of 327 s (one block
  graph for all blocks). Phase 1's MoE runs train under FSDP2.
- Granularity 8 costs 30 ms per step more than granularity 2 at the same parameters (DDP), and exceeds the 74 GiB mark
  under DDP.
- Token choice at capacity 1.5 balances with the bias rule over the first 400 steps: worst layer MaxVio 4.65 -> 1.07,
  dropped assignments 31.9% -> 8.8%.
- Resume (DDP, G0 e): all states restored; the 36 routing buffers are saved, set, measured and mirrored into the EMA
  slot; no statistics are saved. An FSDP2 checkpoint equals a DDP one in keys, shapes and dtypes (478 entries).
- A DDP arm compiles 4-5.5 min with the GPUs at 0%; a second compile of the same graph in the same job took 53 s
  (Inductor cache in the job's TMPDIR).

G0 f (step time <= 1.15 x dense at X 8, G 2) fails: 1.33 x under FSDP2. Next: the step profile w2686
(`tmp/ladder_prof/moe/prof_moe_step.sh`: dense DDP and FSDP2, ecb16 FSDP2, DDP, DDP with the Triton sums) splits the
overhead before any change.
