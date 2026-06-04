#!/bin/bash

# ── Define your prediction directories here ───────────────────────────────────
PRED_DIRS=(
    "path_to_your_predictions/exp_jit_t2i_sampling_gpic_full_390600_cfg_1.75_euler_50_steps/val/predict/"

)

# ── Config ────────────────────────────────────────────────────────────────────
REF_NPZ="path_to_your_data/gpic_test_ref.npz"
BATCH_SIZE=512
MODELS="dino"
LOG_DIR="./eval_logs"

mkdir -p "$LOG_DIR"


# ── Run ───────────────────────────────────────────────────────────────────────
for PRED_DIR in "${PRED_DIRS[@]}"; do
    RUN_NAME=$(echo "$PRED_DIR" | tr -s '/' | sed 's|/val/predict[/]*$||' | xargs basename)
    LOG_FILE="${LOG_DIR}/${RUN_NAME}.txt"

    echo "Running eval for: $PRED_DIR"
    echo "Logging to: $LOG_FILE"

    gpic-eval "$PRED_DIR" "$REF_NPZ" \
        --batch_size "$BATCH_SIZE" \
        --models "$MODELS" \
        2>&1 | tee "$LOG_FILE"

    echo "Done: $PRED_DIR"
    echo "---"
done