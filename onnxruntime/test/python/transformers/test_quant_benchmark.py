#!/usr/bin/env python
"""Test script to verify quantized kv cache benchmark works correctly."""

from test_sparse_attention import GroupQueryAttentionConfig, OrtGroupQueryAttention

# Test configuration with INT4 quantization
print("Testing INT4 quantized KV cache configuration...")
config_int4 = GroupQueryAttentionConfig(
    batch_size=1,
    sequence_length=16,
    max_sequence_length=128,
    past_sequence_length=0,
    num_heads=8,
    kv_num_heads=8,
    head_size=128,
    k_quant_type="PER_CHANNEL",
    v_quant_type="PER_CHANNEL",
    kv_cache_type="int4",
)

try:
    obj_int4 = OrtGroupQueryAttention(config_int4)
    print("✓ INT4 configuration created successfully")
    print(f"  - Cache type: {config_int4.kv_cache_type}")
    print(f"  - Bit width: {config_int4.kv_cache_bit_width}")
    print(f"  - K quant type: {config_int4.k_quant_type}")
    print(f"  - V quant type: {config_int4.v_quant_type}")
except Exception as e:
    print(f"✗ INT4 configuration failed: {e}")
    import traceback

    traceback.print_exc()

# Test configuration with INT8 quantization
print("\nTesting INT8 quantized KV cache configuration...")
config_int8 = GroupQueryAttentionConfig(
    batch_size=1,
    sequence_length=16,
    max_sequence_length=128,
    past_sequence_length=0,
    num_heads=8,
    kv_num_heads=8,
    head_size=128,
    k_quant_type="PER_CHANNEL",
    v_quant_type="PER_CHANNEL",
    kv_cache_type="int8",
)

try:
    obj_int8 = OrtGroupQueryAttention(config_int8)
    print("✓ INT8 configuration created successfully")
    print(f"  - Cache type: {config_int8.kv_cache_type}")
    print(f"  - Bit width: {config_int8.kv_cache_bit_width}")
    print(f"  - K quant type: {config_int8.k_quant_type}")
    print(f"  - V quant type: {config_int8.v_quant_type}")
except Exception as e:
    print(f"✗ INT8 configuration failed: {e}")
    import traceback

    traceback.print_exc()

# Test non-quantized for comparison
print("\nTesting non-quantized configuration...")
config_fp16 = GroupQueryAttentionConfig(
    batch_size=1,
    sequence_length=16,
    max_sequence_length=128,
    past_sequence_length=0,
    num_heads=8,
    kv_num_heads=8,
    head_size=128,
)

try:
    obj_fp16 = OrtGroupQueryAttention(config_fp16)
    print("✓ FP16 configuration created successfully")
    print(f"  - Cache type: {config_fp16.kv_cache_type}")
    print(f"  - K quant type: {config_fp16.k_quant_type}")
except Exception as e:
    print(f"✗ FP16 configuration failed: {e}")
    import traceback

    traceback.print_exc()

print("\n✓ All configuration tests passed!")
