# The final ladder of the post "How to Scale Text-to-image Models: a Handbook": one line per size, the values of the post's seven runs
# (W&B reself/gpic-speed), then the next three rungs d24, d26 and d28 (issues #1, #2 and #3: the same rules at 50 image tokens per image
# parameter, trained across nodes at a larger batch so that each run ends within a day; both learning rates are already multiplied by
# sqrt(batch / 1024)). Sourced by jobs/final_ladder.sh and jobs/eval_final_ladder.sh.
# final_ladder_size <size> sets D (width), NB (image layers), NT (text layers), S (steps), MLR (Muon lr), ALR (AdamW lr), WDK (Muon weight
# decay), MB (images per GPU per pass), GB (global batch in images) and NG (the GPUs the batch is spread over); it returns 2 for an unknown size.
FINAL_LADDER_SIZES="d10 d11 d12 d13 d14 d16 d18 d24 d26 d28"
final_ladder_size() {
  GB=1024 NG=8   # the post's seven runs: batch 1024 on one 8-GPU node
  case $1 in   #  width   image layers  text layers  steps     Muon lr      AdamW lr     Muon decay    per GPU     (FLOPs, run)
    d10) D=768;  NB=10; NT=2; S=21248;  MLR=1.36e-2;  ALR=5.01e-4; WDK=0.0173;  MB=128 ;;   # 3.6e18 xl-d10-0p25c0
    d11) D=896;  NB=11; NT=2; S=28800;  MLR=9.617e-3; ALR=4.06e-4; WDK=0.01805; MB=128 ;;   # 7.1e18 cd-d11-l50-x1
    d12) D=1024; NB=12; NT=3; S=40192;  MLR=6.8e-3;   ALR=3.26e-4; WDK=0.01829; MB=128 ;;   # 1.4e19 xl-d12-1c0
    d13) D=1152; NB=13; NT=3; S=59136;  MLR=4.808e-3; ALR=2.77e-4; WDK=0.01758; MB=128 ;;   # 2.9e19 xl-d13-2c0
    d14) D=1280; NB=14; NT=3; S=89600;  MLR=3.4e-3;   ALR=2.38e-4; WDK=0.01641; MB=128 ;;   # 5.7e19 xl-d14-4c0
    d16) D=1536; NB=16; NT=4; S=108800; MLR=2.404e-3; ALR=1.79e-4; WDK=0.01912; MB=128 ;;   # 1.1e20 xl-d16-8c0
    d18) D=1792; NB=18; NT=4; S=143872; MLR=1.7e-3;   ALR=1.44e-4; WDK=0.02044; MB=64  ;;   # 2.3e20 xl-d18-16c0
    d24) D=2560; NB=24; NT=5; S=270016; MLR=7.56e-4;  ALR=1.18e-4; WDK=0.0245;  MB=16; GB=2048; NG=128 ;;   # 2.3e21 issue #1
    d26) D=2816; NB=26; NT=6; S=176960; MLR=8.15e-4;  ALR=1.43e-4; WDK=0.0347;  MB=16; GB=4096; NG=256 ;;   # 4e21 issue #2
    d28) D=3072; NB=28; NT=6; S=151200; MLR=7.80e-4;  ALR=1.54e-4; WDK=0.0424;  MB=16; GB=6144; NG=384 ;;   # 6.5e21 issue #3
    *) return 2 ;;
  esac
}
