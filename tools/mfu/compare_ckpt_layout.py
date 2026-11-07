#!/usr/bin/env python3
"""Freeze check (a): a checkpoint written with the MFU levers has the layout of one written by the frozen ladder code
(5cd3372): the same top-level keys, the same state_dict keys, shapes and dtypes (the evaluation reads ema_denoiser.* with
strict=False, so a missing or reshaped key would give a wrong FD and no error), the shard clock. Values differ (other runs).
usage: python compare_ckpt_layout.py <reference .ckpt> <new .ckpt> [...pairs]"""
import sys
import torch


def main():
    args = sys.argv[1:]
    ok = True
    for ref, new in zip(args[::2], args[1::2]):
        a = torch.load(ref, map_location="cpu", mmap=True, weights_only=False)
        b = torch.load(new, map_location="cpu", mmap=True, weights_only=False)
        top = (set(a) == set(b))
        sa, sb = a["state_dict"], b["state_dict"]
        keys = set(sa) == set(sb)
        bad = [k for k in sa if k in sb and (tuple(sa[k].shape) != tuple(sb[k].shape) or sa[k].dtype != sb[k].dtype)]
        clock = ("gpic_shard_clock" in a) == ("gpic_shard_clock" in b)
        good = top and keys and not bad and clock
        ok &= good
        print(f"{new.split('/')[-1]} vs {ref.split('/')[-1]}: top-level keys same {top} {sorted(set(a) ^ set(b))}; "
              f"state_dict {len(sa)}/{len(sb)} keys, same {keys}, missing {sorted(set(sa) - set(sb))[:3]}, extra "
              f"{sorted(set(sb) - set(sa))[:3]}, shape/dtype mismatches {bad[:3]}; shard clock alike {clock}: "
              f"{'PASS' if good else 'FAIL'}")
    print("LAYOUT " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
