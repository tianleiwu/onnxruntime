#!/bin/bash

set -e
set -o pipefail

# Parse arguments
RUN_FP16=false
RUN_INT8=false
RUN_INT4=false
RUN_INT8_QUANT=false
RUN_BF16=false

# Profile parameters to pass through to profile_gqa.py
BATCH_SIZE=""
SEQUENCE_LENGTH=""
PAST_SEQUENCE_LENGTH=""
PACKED_QKV=""
SHARE_KV_SCALE=""
NUM_HEADS=""
KV_NUM_HEADS=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --fp16)
            RUN_FP16=true
            echo "==== 🚀 FP16 run enabled ===="
            ;;
        --int8)
            RUN_INT8=true
            echo "==== 🚀 INT8 run enabled ===="
            ;;
        --int4)
            RUN_INT4=true
            echo "==== 🚀 INT4 run enabled ===="
            ;;
        --int8_quant)
            RUN_INT8_QUANT=true
            echo "==== 🚀 INT8 Quant run enabled ===="
            ;;
        --bf16)
            RUN_BF16=true
            echo "==== 🚀 BF16 run enabled ===="
            ;;
        --all)
            RUN_FP16=true
            RUN_INT8=true
            RUN_INT4=true
            RUN_INT8_QUANT=true
            RUN_BF16=true
            echo "==== 🚀 All runs enabled ===="
            ;;
        -b|--batch-size)
            BATCH_SIZE="--batch-size $2"
            echo "==== Batch size: $2 ===="
            shift
            ;;
        -s|--sequence-length)
            SEQUENCE_LENGTH="--sequence-length $2"
            echo "==== Sequence length: $2 ===="
            shift
            ;;
        -p|--past-sequence-length)
            PAST_SEQUENCE_LENGTH="--past-sequence-length $2"
            echo "==== Past sequence length: $2 ===="
            shift
            ;;
        --qkv)
            PACKED_QKV="--is-packed-qkv"
            echo "==== Packed QKV enabled ===="
            ;;
        --share-kv-scale)
            SHARE_KV_SCALE="--share-kv-scale"
            echo "==== Share KV scale enabled ===="
            ;;
        --num-heads)
            NUM_HEADS="--num-heads $2"
            echo "==== Num Heads: $2 ===="
            shift
            ;;
        --kv-num-heads)
            KV_NUM_HEADS="--kv-num-heads $2"
            echo "==== KV Num Heads: $2 ===="
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

# Build extra args string
EXTRA_ARGS="${BATCH_SIZE} ${SEQUENCE_LENGTH} ${PAST_SEQUENCE_LENGTH} ${PACKED_QKV} ${SHARE_KV_SCALE} ${NUM_HEADS} ${KV_NUM_HEADS}"

pip install nvtx

if [ "$RUN_FP16" = true ]; then
    rm -f gqa_fp16.nsys-rep
    rm -f gqa_fp16.sqlite
    nsys profile -o gqa_fp16 --export=sqlite python profile_gqa.py --mode fp16 --warmup 5 --repeat 100 $EXTRA_ARGS
    python parse_nsys.py gqa_fp16.sqlite --skip-first 5 --tag Fp16
fi

if [ "$RUN_BF16" = true ]; then
    rm -f gqa_bf16.nsys-rep
    rm -f gqa_bf16.sqlite
    nsys profile -o gqa_bf16 --export=sqlite python profile_gqa.py --mode bf16 --warmup 5 --repeat 100 $EXTRA_ARGS
    python parse_nsys.py gqa_bf16.sqlite --skip-first 5 --tag Bf16
fi

if [ "$RUN_INT8" = true ]; then
    rm -f gqa_int8.nsys-rep
    rm -f gqa_int8.sqlite
    nsys profile -e ORT_FLASH_ATTENTION_QUERY_DYNAMIC_QUANT=0 -o gqa_int8 --export=sqlite python profile_gqa.py --mode int8 --warmup 5 --repeat 100 $EXTRA_ARGS
    python parse_nsys.py gqa_int8.sqlite --skip-first 5 --tag Int8
fi

# if [ "$RUN_INT8_QUANT" = true ]; then
#     rm -f gqa_int8_quant.nsys-rep
#     rm -f gqa_int8_quant.sqlite
#     nsys profile -e ORT_FLASH_ATTENTION_QUERY_DYNAMIC_QUANT=1 -o gqa_int8_quant --export=sqlite python profile_gqa.py --mode int8 --warmup 5 --repeat 100 $EXTRA_ARGS
#     python parse_nsys.py gqa_int8_quant.sqlite --skip-first 5 --tag Int8Q
# fi

if [ "$RUN_INT4" = true ]; then
    rm -f gqa_int4.nsys-rep
    rm -f gqa_int4.sqlite
    nsys profile -o gqa_int4 --export=sqlite python profile_gqa.py --mode int4 --warmup 5 --repeat 100 $EXTRA_ARGS
    python parse_nsys.py gqa_int4.sqlite --skip-first 5 --tag Int4
fi
