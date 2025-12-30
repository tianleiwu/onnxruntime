
# MoE and QMoE Technical Documentation

This document describes the technical implementation details of the Mixture of Experts (MoE) and Quantized Mixture of Experts (QMoE) operators in ONNX Runtime, specifically targeting the CUDA execution provider.

## 1. Data Layouts

The implementation relies on strict memory layouts for weights and quantization parameters to interact correctly with the underlying Cutlass GEMM kernels.

### 1.1 Weights (QMoE)
For Group-wise Quantized MoE (e.g., 4-bit or 8-bit), the weights are **not** standard linear layers. They must be pre-packed to match the specific interleaving requirements of the Cutlass Mixed Input GEMM kernel.

*   **Logical Shape**: `[NumExperts, HiddenSize, InterSize]` (Note: ONNX uses `[E, In, Out]` convention for MoE, but PyTorch Linear is `[Out, In]`. Effectively `[E, K, N]`).
*   **Physical Layout**: The weights are packed into `uint8` tensors (even for 4-bit, where two 4-bit elements are packed into one byte).
*   **Packing Tool**: The weights are packed using `pack_weights_for_cuda_mixed_gemm` (exposed via `onnxruntime.quantization.matmul_4bits`). This function rearranges the elements into the column-major interleaved format expected by the Cutlass kernel.
*   **Input Index**:
    *   `fc1_experts_weights`: Input 2
    *   `fc2_experts_weights`: Input 5
    *   `fc3_experts_weights`: Input 8 (Optional)

### 1.2 Scales (QMoE)
Scaling factors for dequantization.

*   **Shape**: `[NumExperts, OutputChannels, InputChannels / BlockSize]`.
    *   Example: For `Hidden=4096`, `Block=64`, `Scale` dim is `4096/64 = 64`.
    *   `fc1_scales` (Hidden -> Inter): `[NumExperts, InterSize, HiddenSize // BlockSize]`.
    *   `fc2_scales` (Inter -> Hidden): `[NumExperts, HiddenSize, InterSize // BlockSize]`.
*   **Type**: `float16` (MLFloat16).
*   **Input Index**: 3, 6, 9.

### 1.3 Zero Points (QMoE)
Zero points for asymmetric quantization.

*   **Shape**: Same as Scales: `[NumExperts, OutputChannels, InputChannels / BlockSize]`.
*   **Type**: `uint8` (TensorProto.UINT8).
*   **Input Index**: 11, 12, 13.
*   **Important**: The runtime treats these as **Additive Biases** during `PrePack`. See Section 2.

## 2. PrePack Transformations

The `QMoE` operator implements a `PrePack` method (`moe_quantization.cc`) to optimize quantization parameters before execution.

### 2.1 Weight Packing
*   **Runtime Action**: None. The `QMoE::PrePack` implementation currently **skips** weights. It assumes that weights provided to the ONNX model have *already* been packed using the offline `pack_weights_for_cuda_mixed_gemm` tool.

### 2.2 Scale & Zero-Point Processing
To maximize performance and compatibility with Cutlass kernels that expect a "Scale + Bias" dequantization formula (`q * scale + bias`), the `PrePack` step performs the following:

1.  **Capture Scales**: Copies `fc_scales` from CPU to GPU (or just retains them if already on GPU).
2.  **Compute Bias**: The Cutlass kernel expects an additive bias, but ONNX provides integer Zero Points (`zp`). The PrePack step pre-calculates this bias:
    ```cpp
    Bias = -ZeroPoint * Scale
    ```
    *   This conversion happens via a CUDA kernel `LaunchQMoEPrePackZP`.
    *   The resulting `Bias` is stored in a `float16` buffer.
    *   This allows the kernel to perform `q * scale + bias` which is mathematically equivalent to `(q - zp) * scale`.

The stored `Bias` buffer allows the operator to effectively ignore the original `uint8` Zero-Point tensor during the time-critical `Compute` phase.

## 3. Cutlass Kernel Expectations

The backend uses `CutlassMoeFCRunner` to dispatch kernels.

### 3.1 Kernel Signature
The kernels are typically instantiated as:
```cpp
MoeGemmRunner<half, uint8_t, half>  // InputType, WeightType, OutputType
```

### 3.2 Layout Requirements
*   **Weights**: Column-Major Interleaved (Packed).
*   **Scales & Bias**: Row-Major `[Output, Input/Block]`. (Note: Since we process one expert at a time or grouped experts, the `NumExperts` dimension is the batch).
*   **Bias Type Constraint**:
    *   For Symmetric Quantization (`Bias=0`), the kernel works seamlessly.
    *   For Asymmetric Quantization, the kernel expects `ElementZero` (the bias type) to match `ElementScale` (`half`).
    *   *Note*: There is a known issue where `DefaultGemmGrouped` may infer `ElementZero` type from `ElementB` (Weights, `uint8`) instead of `ElementScale`, which can cause asymmetric parity issues if not explicitly specialized.

## 4. Testing Infrastructure

The primary test scripts are located in `onnxruntime/test/python/transformers/`.

### 4.1 Test Scripts
1.  **`test_qmoe_cuda.py`**:
    *   **Purpose**: Tests the Quantized MoE (QMoE) operator.
    *   **Features**:
        *   Generates random data and quantizes it using `quant_dequant_blockwise`/`quantize_matmul_4bits`/`8bits`.
        *   Constructs an ONNX graph with `QMoE` node.
        *   Packs weights using `pack_weights_for_cuda_mixed_gemm`.
        *   Compares ORT output against a PyTorch reference implementation (Float16 matmul using dequantized weights).
    *   **Usage**:
        ```bash
        python onnxruntime/test/python/transformers/test_qmoe_cuda.py
        # Run specific test
        python onnxruntime/test/python/transformers/test_qmoe_cuda.py -k "test_swiglu_qmoe_blockwise_parity_cpu_4"
        ```

2.  **`test_moe_cuda.py`**:
    *   **Purpose**: Tests the standard (non-quantized) `MoE` operator.
    *   **Features**: Tests FP16/BF16 functionality, routing logic, and standard GEMM parity.

### 4.2 Quantization & Reference
*   **Quantization**: Done in Python using numpy/torch helpers (e.g., `_quantize.quantize_matmul_4bits`).
*   **Reference**: The "Ground Truth" is calculated by dequantizing the weights back to FP16 in Python:
    ```python
    dequantized = (q_weight - zero_point) * scale
    reference = input @ dequantized.T
    ```
    This ensures we verify the numerical correctness of the *dequantization* fusion.

## 5. Technical Details & Nuances

### 5.1 Architecture Support
*   **SM80+ (Ampere)**: The QMoE kernels (Cutlass 3.x / Mixed Input GEMM) primarily target Ampere (SM80) and newer architectures (Hopper SM90).
*   **Tensor Cores**: The packed layout is specifically designed to feed Tensor Cores efficiently.

### 5.2 SwiGLU Fusion
*   **Interleaved**: The operator supports `swiglu_interleaved=1`. In this mode, the weights for the Gating and Value projections are interleaved in the `fc1` tensor.
    *   Shape: `[Experts, 2 * InterSize, HiddenSize]`.
    *   The kernel computes the GEMM, then applies SwiGLU activation + gating in the epilogue.

### 5.3 Memory Management
*   **Workspace**: The operator requires a workspace for intermediate results (sorting indices, permuted rows).
*   **Pre-allocated Buffers**: `PrePack` allocates GPU memory for Scales and Biases. These persist for the lifetime of the session, reducing overhead per inference step.

## 6. Weight Packing Details (`pack_weights_for_cuda_mixed_gemm`)

The `pack_weights_for_cuda_mixed_gemm` function is a critical offline preprocessing step required to format weights for Cutlass Mixed Input GEMM kernels. The source code is distributed across:
*   **Python Binding**: `onnxruntime/python/onnxruntime_pybind_quant.cc`
*   **Kernel Adaptor**: `onnxruntime/contrib_ops/cuda/llm/fpA_intB_gemm_adaptor.cu`
*   **Implementation**: `onnxruntime/contrib_ops/cuda/llm/fpA_intB_gemm_preprocessors_impl.cu` (and `.h`)

### 6.1 Processing Pipeline
The function transforms standard linear weights into a hardware-optimized format through the following stages:

1.  **Input Layout**: Accepts weights in `[N, K]` layout, corresponding to `(Out_Features, In_Features)`.
    *   For 4-bit, data is packed (2 elements per byte).
2.  **Transpose & Signed Conversion**:
    *   Transposes to `[K, N]` layout.
    *   Converts unsigned data to **Signed Int8** intermediate representation.
        *   **4-bit**: Unpacks `uint4` [0, 15], subtracts 8 -> `int8` [-8, 7].
        *   **8-bit**: Subtracts 128 from `uint8` [0, 255] -> `int8` [-128, 127].
3.  **Row Permutation (LDSM Optimization)** (SM80+):
    *   Reorders rows within small tiles to align with the **Load Shared Memory (LDSM)** instruction requirements of Ampere+ Tensor Cores.
    *   **W8_A16 (8-bit)**: Permutes every 16 rows using map `{0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15}`.
    *   **W4_A16 (4-bit)**: Permutes every 32 rows using map `{0, 1, 8, 9, 16, 17...}`.
    *   *Source*: `kPerm_W8_A16` / `kPerm_W4_A16` in `fpA_intB_gemm_preprocessors_impl.h`.
4.  **Interleaving**:
    *   Applies column interleaving (e.g., `ColumnMajorTileInterleave`) if required by the specific Cutlass kernel layout for the target architecture.
5.  **Final Register Layout Adjustment**:
    *   **Add Bias**: Adds 128 to shift values back to alignment with `uint8` storage if needed (effectively negating the earlier subtraction for storage, or adjusting for kernel expectations).
    *   **Swap**: Performs sub-register swapping (e.g., `[0, 1, 2, 3] -> [0, 2, 1, 3]`) for specific register file layouts.
    *   *Kernel*: `add_bias_and_interleave_int8s_inplace_kernel`.

### 6.2 Interleaving Details
The packing process applies specific interleaving patterns to align data with Cutlass kernel expectations (`ColumnMajorTileInterleave`). The parameters depend on the quantization type:

| Quantization Type | Cutlass Kernel Layout | Interleaving Parameters |
| :--- | :--- | :--- |
| **W8_A16** (8-bit) | `ColumnMajorTileInterleave<64, 2>` | `RowsPerTile=64`, `ColumnsInterleaved=2` |
| **W4_A16** (4-bit) | `ColumnMajorTileInterleave<64, 4>` | `RowsPerTile=64`, `ColumnsInterleaved=4` |

*Note*: `RowsPerTile` refers to the K-dimension tile size. `ColumnsInterleaved` indicates how many N-dimension columns are interleaved together.

### 6.3 Cross-Architecture Compatibility
While weight packing is architecture-aware, many architectures share the same layout format. The following table summarizes compatibility:

| Target Architecture | Compatible Packed Weights From... | details |
| :--- | :--- | :--- |
| **SM70 (Volta)* | *Not Supported* | (Requires Tensor Cores with Int8 support in this specific layout) |
| **SM75 (Turing)** | SM75, SM80, SM86, SM89, SM100+ | Uses LDSM Permutation + Interleaving. |
| **SM80 (Ampere)** | SM75, SM80, SM86, SM89, SM100+ | Same as above. |
| **SM86/89 (Ada/Lovelace)** | SM75, SM80, SM86, SM89, SM100+ | Same as above. |
| **SM90 (Hopper)** | **SM90 Only** | **Incompatible**. SM90 skips the interleaving step (`arch != 90`), using a Permuted-Linear layout. |
| **SM100/120 (Blackwell)** | SM75, SM80, SM86, SM89, SM100+ | Falls back to SM80 behavior (LDSM Permutation + Interleaving). |

**Summary**:
*   **Group A (Universal)**: SM75, SM80, SM86, SM89, SM120. Weights packed on any of these can be used on any other in this group.
*   **Group B (Hopper)**: SM90. Weights packed for SM90 are unique to SM90.
