# TensorRT-LLM MoE Plugin: INT4 Group-wise Quantization Format

This document describes the tensor formats for **float16 input with INT4 group-wise quantization** in the TensorRT-LLM Mixture of Experts (MoE) plugin.

## GEMM Dimensions

For MoE GEMM1 (first fully-connected layer):

| Dimension | Description | Example |
|-----------|-------------|---------|
| **M** | Number of tokens assigned to expert | Dynamic |
| **K** | `hidden_size` (input dimension) | 4096 |
| **N** | `inter_size × 2` (for gated activations like SwiGLU) | 28672 |

---

## Weight Tensor

### Logical Shape
```
[num_experts, K, N]
```
Example: `[8, 4096, 28672]` for 8 experts

### Packed Storage Shape
INT4 weights are packed with **2 elements per byte** (8 elements per 32-bit word):

```
[num_experts, K, N // 2]  # in bytes
```

Or equivalently:
```
[num_experts, K, N // 8]  # in int32 words
```

Example: `[8, 4096, 3584]` for `N=28672` (stored as int8 view)

### Storage Format
Weights are stored as packed **int8** (2 INT4 values per byte), viewed as **float16** for TensorRT:

```python
# num_weights_in_32_bits = 8 for INT4
unprocessed_int_weight = torch.randint(..., (num_experts, K, N * 2 // 8), dtype=torch.int32)
unprocessed_weight = unprocessed_int_weight.view(torch.int8)

cuda_q_weight = preprocessor(unprocessed_weight, torch.quint4x2, torch.float16).view(torch.float16)
```

### INT4 Packing Layout (within a byte)
```
Byte: [high_nibble (4 bits) | low_nibble (4 bits)]
      [   element_1        |    element_0       ]
```

Each INT4 element is in range `[-8, 7]` (signed) or `[0, 15]` (after bias).

---

## Preprocessing Pipeline

The weights undergo several transformations in `cutlass_preprocessors.cpp`:

### 1. Row Permutation (for LDSM on Turing+)

For INT4, groups of **32 rows** are permuted using:
```
[0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27,
 4, 5, 12, 13, 20, 21, 28, 29, 6, 7, 14, 15, 22, 23, 30, 31]
```

> [!NOTE]
> This differs from INT8 which uses a 16-row permutation pattern.

### 2. Subbyte Transpose

Row-major `[K, N]` → Column-major `[N, K]` with nibble-level transposition:
- Each INT4 element is individually transposed
- Requires careful bit manipulation to handle nibble swapping

### 3. Column Interleaving (Ampere/sm80)

Uses `ColumnMajorTileInterleave` layout for efficient tensor core access.

### 4. Bias Addition + Register Interleaving

**Bias**: +8 added to each INT4 element
- Shifts signed `[-8, 7]` → unsigned `[0, 15]`

**Register layout transformation**:
```
Input:  [elt_7  elt_6  elt_5  elt_4  elt_3  elt_2  elt_1  elt_0]
Output: [elt_7  elt_5  elt_3  elt_1  elt_6  elt_4  elt_2  elt_0]
```

This interleaving minimizes shift/mask operations in the GEMM kernel.

---

## Scale Tensor

### Shape
```
[num_experts, K // group_size, N]
```
Example: `[8, 32, 28672]` for `group_size=128`

### Memory Layout
- **Row-major** (standard contiguous layout)
- Per-group, per-output-channel scaling

### Data Type
| Mode | Scale Dtype |
|------|-------------|
| W4A16 (FP16 activation) | `float16` or `bfloat16` |
| W4A8 (FP8 activation) | `float16` (always) |

### Special Case: Hopper W4A8

On Hopper (sm90) with W4A8, scales require additional interleaving:
```python
def interleave_scales(scales: torch.Tensor, interleave_dim: int):
    # [E, num_groups, N] --> [E, num_groups // I, N * I]
    E, G, C = scales.shape
    I = get_weight_scale_interleave_factor(interleave_dim, group_size)
    scales_interleaved = scales.reshape(E, G // I, I, C)
    scales_interleaved = scales_interleaved.permute(0, 1, 3, 2)
    return scales_interleaved.reshape(E, G // I, C * I)

# On Hopper, scales are also converted to bfloat16
scale = scale.to(torch.bfloat16).view(activation_dtype)
```

---

## Zero-Point Tensor

### Shape
```
[num_experts, K // group_size, N]
```
Same shape as scale tensor.

### Memory Layout
- **Row-major** (matches scale layout)
- Per-group, per-output-channel zero-point

### Data Type
- Same as scale (`float16` or `bfloat16`)

> [!WARNING]
> On Hopper with W4A8, zero-point tensors are **not supported**:
> ```python
> if get_sm_version() == 90 and has_alpha:
>     if has_zero:
>         pytest.skip("has_zero is not supported in Hopper with WINT4AFP8.")
> ```

---

## Quantization Modes

| Mode | Zero-Point | Flag | Use Case |
|------|------------|------|----------|
| **Symmetric** | Hard-coded +8 bias | Default | Simple weight-only quant |
| **Asymmetric** | Custom tensor | `GroupwiseQuantAlgo::ZERO` | AWQ, GPTQ |

### GroupwiseQuantAlgo Flags
```python
from tensorrt_llm.quantization import GroupwiseQuantAlgo

quant_algo = (
    GroupwiseQuantAlgo.PRE_QUANT_SCALE * has_pre_quant +  # Activation pre-scaling
    GroupwiseQuantAlgo.ZERO * has_zero +                   # Zero-point tensor
    GroupwiseQuantAlgo.BIAS * has_bias +                   # Output bias
    GroupwiseQuantAlgo.W4A8_ALPHA * use_w4a8_awq           # FP8 activation scaling
)
```

---

## Dequantization Formula

### Symmetric Mode (default)
```cpp
// INT4 stored as unsigned [0, 15] after +8 bias
float dequant_weight = (float)(quant_weight - 8) * scale;
```

### Asymmetric Mode (with zero tensor)
```cpp
float dequant_weight = (float)quant_weight * scale + zero;
```

### Python Reference
```python
# Unpack INT4 to INT8 for reference computation
unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
ref_q_weight = unpacker(packed_weight.cpu())  # Returns values in [-8, 7]

# Broadcast scale from [E, num_groups, N] to [E, K, N]
scale_expanded = scale.repeat_interleave(group_size, dim=1)[:, :K, :]

# Dequantize
dequant_weight = ref_q_weight.float() * scale_expanded

if has_zero:
    zero_expanded = zero.repeat_interleave(group_size, dim=1)[:, :K, :]
    dequant_weight += zero_expanded
```

---

## Comparison: INT4 vs INT8

| Aspect | INT4 | INT8 |
|--------|------|------|
| **Elements per byte** | 2 | 1 |
| **Elements per int32** | 8 | 4 |
| **Value range (signed)** | [-8, 7] | [-128, 127] |
| **Bias offset** | +8 | +128 |
| **Row permutation size** | 32 rows | 16 rows |
| **Packed shape** | `[E, K, N//2]` | `[E, K, N]` |

---

## Summary Table

| Tensor | Shape | Data Type | Layout |
|--------|-------|-----------|--------|
| **Input Activation** | `[M, K]` | float16 / bfloat16 | Row-major |
| **Weight** | `[E, K, N//2]` | int8 (packed int4, viewed as fp16) | Preprocessed CUTLASS layout |
| **Scale** | `[E, K // group_size, N]` | float16 / bfloat16 | Row-major |
| **Zero-Point** | `[E, K // group_size, N]` | float16 / bfloat16 | Row-major |
| **Output** | `[M, N]` | float16 / bfloat16 | Row-major |

Where:
- `E` = num_experts
- `K` = hidden_size
- `N` = inter_size × 2 (for gated activation) or inter_size
- `M` = number of tokens
- `group_size` = typically 64 or 128

---

## Supported Group Sizes

| Architecture | Activation | Supported Group Sizes |
|--------------|------------|----------------------|
| Ampere (sm80) | FP16/BF16 | 64, 128 (fixed) |
| Hopper (sm90) | FP16/BF16 | Any multiple of 64 |
| Hopper (sm90) | FP8 | Any multiple of 128 |

---

## Code References

- Weight preprocessing: [`cutlass_preprocessors.cpp`](file:///home/tlwu/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.cpp)
- INT4 unpacker: `torch.ops.trtllm.unpack_int4_packed_tensor_to_int8`
- QuantParams struct: [`moe_kernels.h`](file:///home/tlwu/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h#L242-L419)
- MoE Plugin: [`mixtureOfExpertsPlugin.h`](file:///home/tlwu/TensorRT-LLM/cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h)
- Test reference: [`test_moe_weight_only_quant_matmul.py`](file:///home/tlwu/TensorRT-LLM/tests/unittest/trt/quantization/test_moe_weight_only_quant_matmul.py)
- Groupwise test: [`test_weight_only_groupwise_quant_matmul.py`](file:///home/tlwu/TensorRT-LLM/tests/unittest/trt/quantization/test_weight_only_groupwise_quant_matmul.py)
