# Review: ladder text cache 3 (decode after grouping), 2026-11-03

VERDICT: NOT READY to submit as written (B1: the smoke runs 151970f in every arm). The dataset change itself has no blocker; fix B1, then submit.

Scope: git diff 151970f HEAD (b6cab1b).

## Findings

### B1 (blocker) jobs/smoke_ladder_cached.sh:23,110-111: both long arms run the old code
`WT=$WC/project/gpic-wt/ladder_cache`, and that worktree is a detached HEAD at 151970f (`git -C ladder_cache rev-parse` = 151970f, the same commit as ladder_cache_old). Arm `d640_long` runs `$WT/jobs/train_ladder.sh` with `LADDER_WT=$WT`, so its code is 151970f too: the A/B compares old with old. The script also calls `$WT/tools/prof/ladder_stages.py` (lines 52, 115), so the new "whole arm" lines never print, and `commit.txt` records 151970f. Fix: before the submit, `git -C ladder_cache checkout --detach b6cab1b` (or point WT at this worktree). Then confirm `commit.txt` says b6cab1b.

The same holds for parts 1-4: arms d640, d1024, d1280, resume, epoch and eval also run 151970f. As written, the smoke gives the new data path no coverage at all.

### S1 (should-fix) gpic_dataset.py:288-297: a systematic decode failure becomes silent repetition
`_finish` catches every `Exception` and returns the previous sample. A per-sample fault (one bad JPEG) is fine. A fault that hits every sample is not: a missing PIL plugin, a bug in `_pre`, or an OOM-like `MemoryError`. Then each worker trains on one image, repeated, with one warning per sample, and the step stays green. Fix: count repeats per worker and raise when they pass a bound, e.g. more than 8 in a row or more than 0.1% of the samples since the worker started.

### S2 (should-fix) jobs/smoke_ladder_cached.sh:88 (drops regex): the new repeat warning is not counted
The data-path check counts `webdataset/handlers\.py|KeyError|Traceback`. The repeat warning comes from `gpic_dataset.py` with the text "repeating the sample before it", so a smoke with repeats still passes. Fix: add `repeating the sample before it` to the regex and report its count.

### S3 (should-fix) tools/prof/ladder_stages.py:31-36: checkpoint saves count as stalls
`train_ladder.sh:27` sets LAST = 32000/32 = 1000. So both long arms save `last.ckpt` at steps 1000, 2000 and 3000. The save runs in a batch-end callback, so it lands between two `t_start` values and counts as a stall over 0.5 s. It also sits 24 steps before the old arm's window reads (1024, 2048, 3072). The A/B stays fair, because both arms save, but "N stalls, X s" mixes the two causes. Fix: report gaps at `step % LAST == 0` on their own line, or exclude them from the stall sum.

### N1 (nit) gpic_dataset.py:294: a failure on a worker's first sample stops the run
`_last_sample` is None only until the first success in each worker process. Persistent workers keep it across passes, so the only exposure is the first sample after each launch (256 workers x 1 sample). The crash goes away on resume, because GPIC_SHARD_SEED reshuffles. The smokes logged 0 decode failures. This is acceptable as written. A cheap fallback: hold the bad sample's text and decode the next good image of the same window.

### N2 (nit) gpic_dataset.py:223-224: `img is None` is dead code
`_has_image` accepts only jpg/jpeg/png/webp, and all 4 are in `wds.autodecode.IMAGE_EXTENSIONS` (checked on vis2). So the handler never returns None for them.

## Checked and found correct
- Alignment (tmp/text_cache_review/group_align3.py, vis2): `_group` with num_workers 1, 3, 32, 33, 64, 256, worker ids {0, 1, n/2, n-1}, and stream lengths 0, 5, 31, 32, 33, 1000, 1024, 1056, 3000, 5000. 0 bad cases: every output is a multiple of 32, no duplicates, each 32-block is sorted (one chunk), at most 31 dropped per window end. `max(1, …)` gives 32 samples when workers exceed chunks. `size` resets on each `iter()`, so every pass staggers again. A worker with no shards yields nothing. `_finish` never drops: it returns or raises. So with the DataLoader's per-worker batching of 32, every batch is one chunk of 32.
- Pixels and RNG: `wds.imagehandler("pil")` is the same handler and the same `convert("RGB")` that `wds.decode("pil")` used, so alpha and palette PNGs give the same pixels. Webdataset lowercases suffixes (`lcase=True`), so dropping the uppercase keys changes nothing. The crop draws from the module `random`, which torch seeds per worker (base_seed + id) as before. `_group` and `wds.shuffle` use their own `Random` instances. Deferring the crop changes only the order of the draws, not their distribution. Across workers and ranks the seeding is the same as before the change.
- Fields: the old decode turned `.json` into a dict. `_has_caption` and `_add_caption` already parse bytes (lines 178-189), and nothing else reads `json` or any image field before `_finish`. Collate reads only `s[0..2]`.
- `_last_sample` aliasing (test): the repeat returns the same tuple. `Normalize(inplace=True)` runs inside `_post` before the tuple is stored. `torch.stack` and `bytearray(...)` in collate copy the data, and changing the stacked batch leaves the stored tensor unchanged. The first-sample failure raises `OSError`. Memory held: one sample per worker.
- Non-group path (lines 369-391) and the val dataset are unchanged. Nothing outside gpic_dataset.py uses `image_u8`, `_crop_uint8` or `_DECODE_PIL`.
- Memory: a window holds about 130 KB of compressed image (1.63 GB / 12639 per shard) plus 180 KB of embedding per sample, about 0.32 GB per worker. The old window held 196 KB uint8 plus 180 KB. Tar bytes are copies, so no stream stays open. Collate runs in the workers, so the main process gains no per-sample work.
- Smoke script: `VAR=… arm` does reach the child in bash (non-POSIX mode). jobs/train_ladder.sh and bench.py are not in the diff, so they are the same in both worktrees. ladder_cache_old has gpu_health.py, bench.py, compile_warmup.py and ladder_stages.py. Time: the earlier arms took about 58 min for parts 1-4. Each long arm is about 5 min of startup plus 3400 x 53 ms = 3 min, under ARM_LIMIT 900 s. The total of about 90 min fits in 02:50.
- ladder_stages.py on runs/gpic-speed/ladder_prof_1102_2209/bench/base_d640/steps_rank0.json: the rows have `t_start`. After skipping 32 rows, 568 rows give 567 gaps: mean 53.2 ms, median 49.1 ms, 0 stalls.
- docs/text_cache.md arithmetic: 256 workers x 6 s / 96 CPUs = 16 s (doc: 17 s). 128 steps x 32 ms = 4.1 s. 13 s / 1024 steps = 12.7 ms, so 32 ms becomes 45 ms. 32 x 5 ms = 160 ms per batch, so 8 ranks x 160/32 = 40 CPUs plus tar work, about 45. All consistent.

## Not tested
- Wall-clock behaviour on 8 H100s, GPFS read time of the staggered windows, and in-order delivery under load (that is the smoke's job, after B1).
- A full 14-pass run. Per-rank batch counts at the end of a pass are uneven as before; this diff does not change them.
- Real corrupt images from the dataset: the bad_keys CSV, and whether any key fails decode at all.

## Fixes after this review
- B1: the worktree ladder_cache moves to the fixed commit before the smoke is submitted again; commit.txt records it.
- S1: 8 failed decodes in a row stop the run. Test: a shard of truncated images raises.
- S2: the smoke's check counts repeated samples. S3: ladder_stages.py reports last.ckpt saves apart from other stalls.
- N2: the dead `img is None` branch is gone. N1 stays as written.
