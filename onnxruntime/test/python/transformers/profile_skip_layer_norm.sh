#!/bin/bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
#
# Profile SkipLayerNormalization CUDA kernel with nsys.
#
# Usage:
#   ./profile_skip_layer_norm.sh --baseline             # Profile current kernel
#   ./profile_skip_layer_norm.sh --shmem                # Profile shmem variant
#   ./profile_skip_layer_norm.sh --all                  # Profile both
#   ./profile_skip_layer_norm.sh --all --hidden-size 10240  # Large hidden size (uses non-vectorized path)
#

set -e
set -o pipefail

RUN_BASELINE=false
RUN_SHMEM=false

# Default parameters
BATCH_SIZE=""
SEQ_LEN=""
HIDDEN_SIZE=""
MODE="--mode fp16"
SIMPLIFIED=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --baseline)
            RUN_BASELINE=true
            echo "==== Baseline kernel enabled ===="
            ;;
        --shmem)
            RUN_SHMEM=true
            echo "==== Shmem kernel enabled ===="
            ;;
        --all)
            RUN_BASELINE=true
            RUN_SHMEM=true
            echo "==== All variants enabled ===="
            ;;
        --batch-size)
            BATCH_SIZE="--batch-size $2"
            echo "==== Batch size: $2 ===="
            shift
            ;;
        --seq-len)
            SEQ_LEN="--seq-len $2"
            echo "==== Sequence length: $2 ===="
            shift
            ;;
        --hidden-size)
            HIDDEN_SIZE="--hidden-size $2"
            echo "==== Hidden size: $2 ===="
            shift
            ;;
        --fp32)
            MODE="--mode fp32"
            echo "==== FP32 mode ===="
            ;;
        --simplified)
            SIMPLIFIED="--simplified"
            echo "==== Simplified LayerNorm ===="
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--baseline] [--shmem] [--all] [--batch-size N] [--seq-len N] [--hidden-size N] [--fp32] [--simplified]"
            exit 1
            ;;
    esac
    shift
done

if [ "$RUN_BASELINE" = false ] && [ "$RUN_SHMEM" = false ]; then
    echo "No variant selected. Use --baseline, --shmem, or --all"
    exit 1
fi

EXTRA_ARGS="${BATCH_SIZE} ${SEQ_LEN} ${HIDDEN_SIZE} ${MODE} ${SIMPLIFIED}"

pip install nvtx 2>/dev/null || true

if [ "$RUN_BASELINE" = true ]; then
    echo ""
    echo "========================================"
    echo "  Profiling: Baseline kernel"
    echo "========================================"
    rm -f sln_baseline.nsys-rep sln_baseline.sqlite
    nsys profile -o sln_baseline --export=sqlite \
        -e ORT_SKIP_LAYER_NORM_USE_SHMEM_KERNEL=0 \
        python profile_skip_layer_norm.py --warmup 5 --repeat 100 $EXTRA_ARGS
    echo ""
    echo "---- Baseline kernel results ----"
    python parse_nsys.py sln_baseline.sqlite --skip-first 5 --tag Baseline
fi

if [ "$RUN_SHMEM" = true ]; then
    echo ""
    echo "========================================"
    echo "  Profiling: Shmem kernel (TRT-LLM style)"
    echo "========================================"
    rm -f sln_shmem.nsys-rep sln_shmem.sqlite
    nsys profile -o sln_shmem --export=sqlite \
        -e ORT_SKIP_LAYER_NORM_USE_SHMEM_KERNEL=1 \
        python profile_skip_layer_norm.py --warmup 5 --repeat 100 $EXTRA_ARGS
    echo ""
    echo "---- Shmem kernel results ----"
    python parse_nsys.py sln_shmem.sqlite --skip-first 5 --tag Shmem
fi

echo ""
echo "Done."
