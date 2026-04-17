#!/usr/bin/env bash
# run_experiments.sh — Execute all PPS experiments on the remote workstation.
#
# Phase 1 (head warm-up, 10 epochs, frozen backbone):
#   Experiments A & B run in parallel, each on 1 GPU.
#   Then C & D in parallel. Then E & F in parallel.
#
# Phase 2 (full fine-tune, 40 epochs):
#   Each experiment runs sequentially using BOTH GPUs (--num-gpus 2).
#   lr is already scaled ×2 in phase2_finetune.py for effective batch_size=16.
#
# Usage:
#   cd /workspace/projects/PTv3_PPS
#   bash run_experiments.sh 2>&1 | tee run_experiments.log

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRAIN="python train.py"
P2_CFG="configs/experiments/phase2_finetune.py"

run_phase1_parallel() {
    local exp1=$1 cfg1=$2 exp2=$3 cfg2=$4
    echo ">>> Phase 1 parallel: $exp1 (GPU 0) | $exp2 (GPU 1)"
    CUDA_VISIBLE_DEVICES=0 $TRAIN --config-file "$cfg1" --num-gpus 1 \
        > "logs/${exp1}_phase1.log" 2>&1 &
    pid1=$!
    CUDA_VISIBLE_DEVICES=1 $TRAIN --config-file "$cfg2" --num-gpus 1 \
        > "logs/${exp2}_phase1.log" 2>&1 &
    pid2=$!
    wait $pid1 && echo "  ✓ $exp1 phase1 done" || echo "  ✗ $exp1 phase1 FAILED"
    wait $pid2 && echo "  ✓ $exp2 phase1 done" || echo "  ✗ $exp2 phase1 FAILED"
}

run_phase2_2gpu() {
    local exp=$1 ckpt=$2 save=$3 extra_opts=${4:-""}
    echo ">>> Phase 2 (2 GPUs): $exp"
    CUDA_VISIBLE_DEVICES=0,1 $TRAIN --config-file "$P2_CFG" --num-gpus 2 \
        --options resume=True \
                  load_from="$ckpt" \
                  save_path="$save" \
                  $extra_opts \
        > "logs/${exp}_phase2.log" 2>&1
    echo "  ✓ $exp phase2 done"
}

mkdir -p logs

# ── Round 1: Full supervision vs Partial supervision ─────────────────────────
run_phase1_parallel \
    "exp_A" "configs/experiments/exp_A_full_sup.py" \
    "exp_B" "configs/experiments/exp_B_partial_sup.py"

run_phase2_2gpu "exp_A" \
    "exp/A_full_sup/phase1/model/model_last.pth" \
    "exp/A_full_sup/phase2"

run_phase2_2gpu "exp_B" \
    "exp/B_partial_sup/phase1/model/model_last.pth" \
    "exp/B_partial_sup/phase2" \
    "model.head.supervised_class_ids=[13,14,18,21,22]"

# ── Round 2: Ablations ───────────────────────────────────────────────────────
run_phase1_parallel \
    "exp_C" "configs/experiments/exp_C_no_ortho.py" \
    "exp_D" "configs/experiments/exp_D_adaptive_ema.py"

run_phase2_2gpu "exp_C" \
    "exp/C_no_ortho/phase1/model/model_last.pth" \
    "exp/C_no_ortho/phase2"

run_phase2_2gpu "exp_D" \
    "exp/D_adaptive_ema/phase1/model/model_last.pth" \
    "exp/D_adaptive_ema/phase2" \
    "model.head.adaptive_ema=True"

# ── Round 3: Novel designs ───────────────────────────────────────────────────
run_phase1_parallel \
    "exp_E" "configs/experiments/exp_E_two_temperature.py" \
    "exp_F" "configs/experiments/exp_F_partial_adaptive.py"

run_phase2_2gpu "exp_E" \
    "exp/E_two_temperature/phase1/model/model_last.pth" \
    "exp/E_two_temperature/phase2" \
    "model.head.rare_temperature=0.04 model.head.rare_class_ids=[13,14,18,21,22]"

run_phase2_2gpu "exp_F" \
    "exp/F_partial_adaptive/phase1/model/model_last.pth" \
    "exp/F_partial_adaptive/phase2" \
    "model.head.supervised_class_ids=[13,14,18,21,22] model.head.adaptive_ema=True"

echo ""
echo "All experiments done. Results in exp/*/phase2/"
echo "Compare val mIoU in logs/*_phase2.log"
