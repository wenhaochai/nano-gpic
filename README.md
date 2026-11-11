# nano-gpic: a compute ladder for pixel-space text-to-image

This repository trains a compute ladder of PixelGen's JiT-T2I on [GPIC](https://gpic.stanford.edu) with 8 NVIDIA H100 GPUs per model. We use the ladder to measure scaling laws for pixel-space text-to-image before testing new methods against them.

The goal:

1. Measure how image quality scales with model size, training compute and unique images, in the units of language-model scaling studies: FLOP budgets, iso-FLOP curves, tokens per parameter, unique and total tokens.
2. Find how many passes over the data cost nothing, before fitting any optimum.
3. Only then test new methods, each against the ladder at the same compute.

This is a fork of the GPIC code release ([keshik6/gpic](https://github.com/keshik6/gpic); Chandrasegaran, Sargent et al., 2026; its README is [GPIC_README.md](GPIC_README.md)). Every run, number and figure is on the experiment page, [wenhaochai.com/exps/gpic-speed.html](https://wenhaochai.com/exps/gpic-speed.html). Training logs are in [W&B reself/gpic-speed](https://wandb.ai/reself/gpic-speed). The speed records, their rules and the speed work are in [docs/speedrun.md](docs/speedrun.md).

*Updated 2026-11-10.*

## 1. The ladder

### Models

- **Model.** PixelGen's JiT-T2I: x-prediction flow matching, 16-pixel patches (256 image tokens per 256 × 256 image), Qwen3-1.7B caption features.
- **Sizes.** A model dN has N image blocks (plus 2 to 4 text blocks). We write a shape as width × (image blocks + text blocks). The ladder runs d9, d10, d11, d12, d14, d16 and d18 (widths 640, 768, 896, 1024, 1280, 1536 and 1792), from 109M to 1.68B parameters. The released GPIC model is d16.
- **Budgets.** 3.6e18, 7.1e18, 1.4e19, 2.9e19, 5.7e19 and 1.1e20 FLOPs. The first Muon ladder trains d9, d10 and d11 to 1.4e19, 2.9e19 and 5.7e19 FLOPs and d12 to d18 to 1.1e20 FLOPs, about 5.6e20 FLOPs in all.
- **Data.** GPIC Full: 99,975,623 training images in 8,000 shards. One pass is 97,632 steps at batch 1024. Every run reads one seeded random permutation of the 8,000 shards, a new one each pass, so a shorter run reads a uniform sample of shards. GPIC Lite has 10.1M images; the repetition arms use the first 80 shards (1.01M images) and the first 8 (101K).
- **Scores.** The held-out loss first: the loss on 3072 held-out images at 16 noise levels with fixed noise. FD-DINOv2 as a reference, with the official evaluation (Euler at 50 steps, 50,000 test captions, one noise for all prompts, no CFG), plus the official Inception metrics (FD, precision, recall, density, coverage, MMD).

### Recipe

- **The first Muon ladder.** MiniT2I's settings (Adam, batch 1024, β2 0.95), turned into the MuP version of Muon by lr × 0.2√d ([Su, 2025](https://kexue.fm/archives/11416)): Muon at 3.4e-3 for the transformer matrices (6.8e-3 from d14), Adam for the rest, 1000 warmup steps and a constant rate after them, EMA 0.9996 per step.
- **The next ladder (no EMA).** The same optimizer split, 1% warmup, a linear cooldown over the last 50% of the steps to an absolute 3e-5, Muon weight decay, the raw weights scored. Its rate, weight decay, shape and tokens per parameter come from the nine steps in part 2. The final configuration is set once the re-done iso-FLOP curves fix the tokens per parameter.

<img src="docs/readme/readme_muon_adamw.png" width="560" alt="FD at 1.4e19 FLOPs by parameters, Muon against AdamW">

The Muon recipe beats the AdamW recipe at every point. On GPIC Full at the same images, Muon scores 14% to 24% lower FD than the official AdamW recipe at all twelve points tried (d9 to d12, 3.6e18 to 1.4e19 FLOPs). This compares recipe against recipe: the batch and the rates change with the optimizer.

### Running it

```bash
# usage: jobs/train_muon_ladder.sh <full|lite> <size> <d> <blocks> <text blocks> <steps to 1.4e19 FLOPs> <multiple of that budget> <Adam lr> <Muon lr>
bash jobs/train_muon_ladder.sh full d640  640  9  2 133600 1 6.4e-4 3.4e-3
bash jobs/train_muon_ladder.sh full d768  768  10 2 85024  2 5.0e-4 3.4e-3
bash jobs/train_muon_ladder.sh full d896  896  11 2 57600  4 4.1e-4 3.4e-3
bash jobs/train_muon_ladder.sh full d1024 1024 12 3 40192  8 3.3e-4 3.4e-3
bash jobs/train_muon_ladder.sh full d1280 1280 14 3 22400  8 2.4e-4 6.8e-3
bash jobs/train_muon_ladder.sh full d1536 1536 16 4 13600  8 1.8e-4 6.8e-3
bash jobs/train_muon_ladder.sh full d1792 1792 18 4 8992   8 1.4e-4 6.8e-3
```

Each line trains one model on one 8×H100 node, resumes from its last checkpoint, and carries the model's speed settings itself (the scripts name a model by its width). Before the first run: GPIC Full (`tools/download_full.py`) and its Qwen3-1.7B caption features (`jobs/encode_captions.sh`, checked by `tools/check_text_cache.py`). The scripts keep absolute paths and a job-queue wrapper, so set them to yours. The no-EMA runs launch from branch `launch-cutfix-cap`: it adds levers that cut the time to a launch's first step, and the cut fix (a cut right after a launch's first save leaves a whole checkpoint, and a failed launch leaves no process).

### Results

<img src="docs/readme/readme_all_points.png" width="560" alt="FD of every evaluation against training compute 6ND: the main-line ladder and the released model">

Every evaluation of every run, against training compute 6ND. Blue is the main line, the Muon ladder on GPIC Full, and the light-blue line follows the lowest FD of each of its budgets. Red is the released GPIC model. Both use the standard evaluation (Euler 50, the official noise, no CFG). Grey is every other evaluation.

<img src="docs/readme/readme_compute.png" width="560" alt="The lowest FD of each budget against training compute">

In the held-out loss, the lowest point of each budget falls from 0.06857 at 3.6e18 FLOPs to 0.06446 at 1.1e20 FLOPs. The best size grows with compute, from d9 to d16. In FD it falls from 979.3 to 338.9 (d16). From 5.7e19 FLOPs on, the ladder beats the released GPIC model with less than half its compute. The released model is d16, AdamW, one pass over Full, 1.4e20 FLOPs; its held-out loss is 0.06536 with its own text path, and its FD 472.4.

### Where the code is

| Branch | Contents |
|---|---|
| `main` | the production line: the ladder (per-model speed settings, Muon also under FSDP2, the shared caption bucket), `jobs/train_muon_ladder.sh`, `configs_h100/muon_ladder.yaml`, and the job scripts of the learning-rate, EMA, caption-bucket and batch experiments |
| `ladder-sweep` | the sweep without the EMA: a linear cooldown to an absolute floor (`WarmupFlatCooldown` min_lr) and Muon weight decay on the ladder's code |
| `sweep-noema`, `sweep-noema-cap` | the same with no EMA update at all (`GPIC_MFU_NO_EMA=1`), and with the shared caption bucket's pools capped |
| `sweep-noema-sched` | the cooldown's shape and length (`WarmupFlatCooldown` shape linear or 1−√): the cooldown sweep and the rate and batch checks at larger compute |
| `resume-fix-cap`, `resume-fix-sched` | a resumed launch draws fresh random streams after its warm-up |
| `save-on-term-cap`, `save-on-term-sched` | the training shell waits on a cut for main.py's save |
| `ckpt-dedup-cap`, `ckpt-dedup-sched` | a checkpoint writes the raw weights once |
| `launch-cutfix-cap` | the code of the runs launched since 2026-11-10: the above on the capped line, the launch levers and the cut fix |
| `heldout-release` | `tools/heldout_release.py`: the released model's held-out loss with its own text path (captions padded to 300 tokens, no mask) |
| `heldout-loss`, `heldout-loss-run5` | `tools/heldout_loss.py`: held-out and training loss at fixed noise, EMA and raw weights; run5 is the deployed version |
| `eval-official`, `eval-official-run10`, `eval-official-run11` | the official evaluation with the Inception metrics on every run (`tools/official_inception.py`); run10 moves the DINOv2 transform to the GPU, bit for bit; run11 is the deployed version |
| `ladder-pnsr`, `ladder-pnsr-ma` | the ladder's later segments: Newton-Schulz in the GEMMs, the GPU resize kernels, and Muon's per-part updates in one launch |
| `mfu` | speed work: fused kernels, FSDP2, packed captions, FP8, GPU decoding; the log is `docs/mfu.md` |
| `muon-search2` | job scripts of the learning-rate, EMA, caption-bucket and batch experiments |
| `eval-par` | `jobs/evals_gmflow.sh --par`: one evaluation per GPU, eight side by side |
| `ladder` | the first ladder: the official AdamW recipe on GPIC Lite, no caption dropout |
| `muon` | the first optimizer ablation: Muon against AdamW at the official recipe |
| `gmflow` | a Gaussian mixture of the velocity at each step (K = 1, 8, 16) |
| `dist-step` | a distribution per step: notes and tests |
| `energy-head` | an energy-score head over the whole image |
| `ar` | autoregressive sampling of the image tokens |
| `implicit-nodrop` | a posterior sampler, and the point model without caption dropout |

## 2. How the ladder is built

Nine steps, in order. Each gives a decision and its evidence.

**1. The recipe without the EMA.** Decision: an optimizer split, 1% warmup, a linear cooldown over the last 50% of the steps, the raw weights scored. Evidence: a 50% cooldown lowers the held-out loss by 0.22% to 0.24% against 20% (sweep CD). The EMA lags in short runs. It starts at the initial weights and keeps 0.9996^step of them. Above a 1% share its held-out loss is 4.5% to 590% above the raw weights'; below 0.2% it is 0.3% to 0.8% lower. Without weight decay, runs with a linear cooldown score 5% to 14% higher FD than the EMA ladder at the same compute.

<img src="docs/readme/readme_ema_lag.png" width="560" alt="EMA over raw held-out loss against the EMA's share of the initial weights">

**2. The learning rate.** Decision: a rate rule by size and compute. Evidence: at 3.6e18 FLOPs the best Muon rate is 1.36e-2 at all four sizes tried, up to 5.44e-2. Sweep X trains each size at three rates around the rule at 3.6e18, 7.1e18 and 1.4e19 FLOPs. At d16 and 5.7e19 FLOPs, each run in one piece, the held-out loss is 0.06501 at 1.7e-3, 0.06493 at the rule's 3.4e-3 and 0.06492 at 6.8e-3 (FD 461.6, 458.6 and 457.5). The rule's rate is within 0.02% of twice its rate; half its rate is 0.12% worse. In the first Muon ladder (EMA, constant rate), the largest rate tried wins in short runs: at 1.4e19 FLOPs, 6.8e-3 beats 3.4e-3 by 5.9% at d14 (22.4k steps) and 3.1% at d12 (40.2k), and ties at d9 (133.6k). Smaller rates lose at every size: at 1.4e19 FLOPs, 1.7e-3 is 2.4%, 2.1% and 11.2% worse than 3.4e-3 at d9, d10 and d12, and 0.85e-3 is 11.6% worse at d9.

**3. Weight decay.** Decision: Muon weight decay, set as the decay timescale 1/(lr·wd) relative to the run. Evidence: at d11 and 7.1e18 FLOPs, 0.05 lowers FD from 906.6 to 804.3 (the EMA ladder: 810.8) and FID from 66.99 to 56.83. Its decay timescale is 0.2 of the run.

<img src="docs/readme/readme_noema.png" width="440" alt="FD of runs without the EMA: by Muon weight decay at d11, and by parameters at 1.4e19 FLOPs">

**4. The shape.** Decision: open. The optimum is not bracketed on the deep side; the ladder's shapes stay in use meanwhile. Evidence (sweep AR, depth against width at a fixed number of parameters): at about 110M parameters (1.5e18 FLOPs), 512 × (15+2) is best, with a held-out loss of 0.06891 against the ladder's 640 × (9+2) at 0.06897. At about 250M (7.9e18 FLOPs) and about 670M (5.4e19 FLOPs), the loss falls toward deeper, narrower shapes through the deepest tested: 640 × (23+2) 0.06679 against the ladder's 896 × (11+2) 0.06695, and 896 × (32+3) 0.06464 against 1280 × (14+3) 0.06492. At equal FLOPs the deepest shapes take 15% and 23% more H100-hours (wall time, launches included).

**5. The batch against the steps.** Decision so far: batch 1024 at every size, as in the first Muon ladder, with the steps of one ladder within a factor of ten. The batch checks at larger compute are not yet results.

**6. Iso-FLOP curves.** Decision: the tokens per parameter of the best size, about 48 image tokens per block parameter so far. Evidence: with each size at its best rate, the best size is 120M, 171M and 248M block parameters at 3.6e18, 7.1e18 and 1.4e19 FLOPs (47 to 49 image tokens per parameter at all three budgets), and grows as compute to the power 0.53. At a fixed rate it had been 58, 36 and 30 tokens per parameter. The iso-FLOP curves at 2.9e19 and 5.7e19 FLOPs come next.

**7. The ladder itself.** Decision: one run per size at that ratio, each with its own cooldown, resumed across launches with fresh random streams. Evidence: a resume before the reseed fix raised FD by 4.8% to 5.6% and left the held-out loss unchanged (within 0.02%). From one saved state, the arm that continues and the arm resumed in a new process reach the same held-out loss and FD 808 against 851. A resumed launch now draws fresh random streams after its warm-up (`resume-fix-cap`, `resume-fix-sched`).

**8. Scoring and fit.** Decision: every rung scored as in part 1, the held-out loss first, and fitted as y = E + a·C^(−b) against the baseline's compute. Evidence: FD needs care. Nine runs of d9 whose held-out loss agrees to 0.07% span FD 949 to 1017. With one caption-length bucket per step, FD depends on the seed (966.6 to 1016.6) and averages 4% above runs with thirty-two buckets per step, which repeat within 0.3%. The ladder keeps the one-bucket setting, the fastest, and reads small FD gaps between single runs from the held-out loss.

**9. Repetition.** Decision: measure how many passes over the unique images cost nothing before fitting any optimum. Budgets above 97,632 steps repeat data; smaller ones do not. Evidence: on the AdamW ladders, GPIC Lite costs nothing up to 17 passes over its 10.1M images: its held-out loss is within −0.12% to +0.07% of Full's at the same model and step. Without the EMA, d10 at 7.1e18 FLOPs scores a lower FD on fewer unique images (101K: 827.5, against 892.9 on Full), which may reward copying the training images; the held-out loss decides.
