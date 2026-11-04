VERDICT: GO. The region crop is correct on every size tested, never raises, and differs by at most 2 of 255. Create the worktree gpic-wt/ladder_crop at 127b1e6 before the submit.

Review of commit 127b1e6 (branch ladder), 2026-11-04. Analysis ran on vis2 (nice 19, OMP_NUM_THREADS=1, no GPU). Scripts: /scratch/gpfs/GROUP/USER/tmp/review_crop/{t_bounds.py,t_run.py,t_ident.py}.

## Findings, most severe first

1. jobs/smoke_ladder_crop.sh:17. The worktree is missing.
   - Failure: `$WC/project/gpic-wt/ladder_crop` does not exist today (`git worktree list` shows no such entry). The job prints "no worktree" and exits 2 after it has waited in the queue.
   - Fix: `git worktree add --detach $WC/project/gpic-wt/ladder_crop 127b1e6` before the submit. Then check `git -C .../ladder_crop rev-parse HEAD`.

2. baselines/PixelGen/src/data/dataset/gpic_dataset.py:24. The log does not show which crop is active.
   - Failure: the print still says "random_crop_arr" in both arms. No line in d640_crop.log proves that the switch reached python. If the switch were lost, the crop arm would quietly be a third control.
   - Fix: print `_train_crop.__name__` in that line, or in the GPICWebDataset init. Then grep the arm logs for it after the run.

3. baselines/PixelGen/src/utils/patch_bugs.py:18 together with gpic_dataset.py:307. The repeat-sample fallback is silent. This is an old issue, not new in this commit.
   - Failure: main.py imports patch_bugs, which ignores every UserWarning. The `warnings.warn(...)` in `_finish` has the category UserWarning. So a crop that raised would repeat the previous sample with no log line, up to 8 times in a row. My tests found no exception (see below), so the risk is low for this commit.
   - Fix: give that warning its own category that the filter does not ignore, or log it with `print(..., flush=True)`. Optional: count the repeats and report the count.

No finding against augmentation.py itself.

## What I checked and found correct

Q1. Pass order (Resample.c, ImagingResampleInner, lines 842-940).
- With scale != 1, the official call always runs both passes. The short side maps exactly to smaller_dim_size, so it changes by at least 1 pixel. For the long side L >= min, |L*s - L| >= |sds - min| >= 1, so round(L*s) != L. So need_horizontal and need_vertical are both true.
- With scale == 1, Python's `resize` returns a copy and no pass runs. The new function then gives the same pixels: 2000 calls at min side = smaller_dim_size, 0 differences (t_ident.py).
- Vertical-first branch: the first call keeps xsize = W with box[0] = 0 and box[2] = W, so only the vertical pass runs. The second call keeps ysize = 256 with box[1] = 0 and box[3] = 256, so only the horizontal pass runs. The order matches the official, whose condition the code copies exactly.
- Horizontal-first branch: the first call keeps the height r1-r0 with box (x0, 0, x1, r1-r0), so only the horizontal pass runs. The second call keeps width 256 with box (0, ., 256, .), so only the vertical pass runs. The integer box edges are exact in float32.
- In both orders, the intermediate image is 8-bit, the same as in the official.

Q2. Row margin.
- Analytic: the official first row for output row cy is int(y0 + 0.5*sy - 2*fs + 0.5), with fs = max(1, sy). This value is at least y0 - 2*fs + 0.5, which is more than int(y0 - support) - 1 = r0. The last row bound is int(y1 - 0.5*sy + 2*fs + 0.5) <= y1 + support + 0.5, which is less than int(y1 + support) + 2 = r1. So the window of every kept row lies inside [r0, r1) with a margin of at least about 1 row. Single-precision error cannot use up that margin.
- So the clip at r0 and r1-r0 in the second call never removes a row that the official uses. The clips at 0 and H fall on the same rows. The normalization sums over the same rows.
- Numeric check (t_bounds.py): all W, H in 1..1200, smaller_dim_size in {256, 288, 320}, crop_y in {0, middle, last} and crop_x in {0, last}. I copied precompute_coeffs (double centers) and compared its bounds with [r0, r1). Result: 0 windows outside. This includes upscale (sy < 1) and downscale up to sy just below 2 (the BOX halving keeps sy < 2).

Q3. Exceptions.
- The C check is box[0] < 0, box[1] < 0, box[2] > xsize, box[3] > ysize, or an empty box. I copied it with float32 rounding of the box and a stricter test (width <= 0). It passed for all 4 boxes per call over the grid above: 0 failures. Also 0 cases where out_h or out_w < 256 (randrange stays valid).
- Real calls, both functions, same seeds: 120,180 calls finished with 0 exceptions in either function. The sets were: L noise, binary, checkerboard, stripes and step images at every W in 1..1200 against 25 heights and the transpose (60,000 calls); 30,000 random sizes each in L and RGB; 90 elongated sizes each in L and RGB (1x1500 up to 6000x257). The RNG state after each call matched the official in every call (the assert never fired).
- Not finished: the RGB copy of the every-W sweep (60,000 calls). It hit the 10 min tool limit and I stopped it. The L copy of the same sweep finished with 0 exceptions. Pixel mode does not change the box math.

Q4. Largest difference on adversarial images.
- Worst found: 2 of 255, on every image set. Examples: a binary image of 128x639 (seed 43), and one of 2000x2. Calls by largest difference over 30,000 random RGB sizes: 0: 13,315, 1: 15,912, 2: 773.
- The share of differing values is higher on adversarial images (0.61% on binary noise, 0.18-0.23% on elongated images) than on real images (0.033-0.035%). The 2 of 255 bound holds for both.
- tools/test_region_crop.py 2000 reproduced the docstring: 0.0334-0.0353% of values differ, largest 2, the same RNG state, all hard shapes run.

Q5. The switch and other paths.
- `GPIC_REGION_CROP` is read once, at module import. The DataLoader workers inherit `_train_crop` (fork) or the environment (spawn). Either way they get the same choice.
- Unset: `_train_crop is random_crop_arr`, with the same arguments. The lambda looks up the global at call time. The default path is byte-for-byte the old one.
- The new function returns a uint8 HWC ndarray, not a PIL image. `_crop_uint8` wraps the result in `np.asarray(..., uint8)`, so the grouped path is the same. The non-grouped path passes the ndarray to ToTensor, which divides uint8 arrays by 255 the same way. lite_ladder.yaml uses group_text_len: true and passes no transform.
- The diff does not touch val_transform, center_crop_arr, `_apply_transform`, the shard-seed path or `_group`.

Q6. jobs/smoke_ladder_crop.sh.
- Its structure matches jobs/smoke_ladder_pin3.sh: the same arm(), d640(), d1280() and wait_arm(), the same limits (300 s, 420 s), the same steps (2400, 800), the same #SBATCH lines and the same 42 min wall time. The arms changed on purpose: control, crop, crop + pin + idle, control2, d1280 control, d1280 crop. The rules in the header cover each arm.
- Time: 4 x 300 s + 2 x 420 s = 34 min, plus at most 60 s of kill grace per arm. This fits 42 min, with the same margin as pin3.
- A prefix such as `GPIC_REGION_CROP=1 d640 ...` exports the variable for the length of the function call, and that includes the nested arm() and the child `bash train_ladder.sh`. I tested this pattern in bash: the child sees the value with the prefix, and the variable is unset without it. train_ladder.sh and env.sh do not unset or overwrite any GPIC_REGION_CROP, GPIC_PROF_PIN or GPIC_WORKER_IDLE value. host_tuning.py reads GPIC_PROF_PIN and GPIC_WORKER_IDLE at the ladder HEAD.
- Arm names in phases.log: wait_arm greps "END <tag> " with a trailing space, so d640_crop does not match d640_crop_pin_idle, and d640_control does not match d640_control2. The names are unique.

## What I did not check

- The speed numbers and the adoption decision. These were ruled out.
- The RGB copy of the every-width sweep. It was stopped at the time limit (see Q3).
- Images larger than 1200 px on both sides. For those, the BOX halving brings the short side below 2 x smaller_dim_size before the bicubic step. The elongated tests covered long sides up to 6000.
- Modes other than L and RGB. `_crop_uint8` always converts to RGB first.
- Any GPU run. No job was submitted.
