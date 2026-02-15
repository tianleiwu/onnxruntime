
# MoE and QMoE Technical Documentation

This document describes the technical implementation details of the Mixture of Experts (MoE) and Quantized Mixture of Experts (QMoE) operators in ONNX Runtime, specifically targeting the CUDA execution provider.

## 1. Data Layouts

The implementation relies on strict memory layouts for weights and quantization parameters to interact correctly with the underlying Cutlass GEMM kernels.

### 1.1 Weights (QMoE)
For Group-wise Quantized MoE (e.g., 4-bit or 8-bit), the weights are **not** standard linear layers. They must be pre-packed to match the specific interleaving requirements of the Cutlass Mixed Input GEMM kernel.

*   **Logical Shape**: `[NumExperts, HiddenSize, InterSize]` (Note: ONNX uses `[E, In, Out]` convention for MoE, but PyTorch Linear is `[Out, In]`. Effectively `[E, K, N]`).
    *   *Clarification*: This shape corresponds to the `MatMulNBits` standard (`[Experts, N, K_Blocks] `) *before* architecture-specific packing.
*   **Physical Layout (Storage)**: Opaque Blob.
    *   The actual stored data is **no longer** in the logical shape. It is a packed, transposed, and interleaved blob formatted specifically for the Cutlass kernel.
    *   **Packing**: The `pack_weights_for_cuda_mixed_gemm` function transforms the logical `[E, N, K]` weights into this architecture-dependent blob.
    *   *Note*: Do not attempt to interpret the raw bytes of this tensor as a standard array; it must be treated as a blob passed directly to the kernel.
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
    *   *Reference*: This layout corresponds to the ONNX interface `[NumExperts, Output, Input_Blocks]`.
    *   *Note*: The underlying Cutlass kernel (and TensorRT-LLM specifications) describes a transposed order `[Groups, N]`. The ONNX Runtime implementation adapts this to ensure the correct kernel execution.
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
To maximize performance and compatibility with Cutlass kernels, the `PrePack` step adapts the data based on the quantization type:

1.  **Capture Scales**: Copies `fc_scales` from CPU to GPU.
2.  **Zero-Point / Bias Conversion**:
    *   **8-bit Weights**: Weights are shifted by -128 (uint8 -> int8). We compute a bias to compensate:
        ```cpp
        Bias = (128 - ZeroPoint) * Scale
        ```
        *   This effectively treats the calculation as `(W_stored + 128 - ZP) * Scale`.
    *   **4-bit Weights**: Zero Points are unpacked from nibbles (2 per byte) and stored as **Unscaled** floating-point values (just the integer ZP cast to float/half).
        *   The Cutlass 4-bit kernel (`FINEGRAINED_SCALE_AND_ZEROS`) natively handles `(W - ZP) * Scale` without requiring a pre-computed bias.
    *   **Symmetric**: Bias is 0.

    *   This conversion happens via generic kernels (`LaunchQMoEPrePackOffsetBias`, `QMoEPrePackPacked4BitZPKernel`).
    *   The resulting buffer (`packed_bias`) is stored in `float16/float`, matching the Scale type.

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
*   **Bias/ZP Type Constraint**:
    *   For Symmetric Quantization (`Bias=0`), the kernel works seamlessly.
    *   For Asymmetric Quantization, the kernel expects `ElementZero` (the bias/ZP buffer type) to match `ElementScale` (`half` or `float`).
        *   **4-bit**: Stores Unscaled ZP values.
        *   **8-bit**: Stores Pre-calculated Bias.
    *   *Note*: `DefaultGemmGrouped` may infer `ElementZero` from `ElementB` (Weights, `uint8`). This implementation ensures `ElementZero` matches `ElementScale` (Floating Point) to support asymmetric quantization correctly.

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
*   **Interleaved**: The operator supports `swiglu_fusion=1`. In this mode, the weights for the Gating and Value projections are interleaved in the `fc1` tensor.
    *   Shape: `[Experts, 2 * InterSize, HiddenSize]`.
    *   The kernel computes the GEMM, then applies SwiGLU activation + gating in the epilogue.

### 5.3 Memory Management
*   **Workspace**: The operator requires a workspace for intermediate results (sorting indices, permuted rows).
*   **Pre-allocated Buffers**: `PrePack` allocates GPU memory for Scales and Biases. These persist for the lifetime of the session, reducing overhead per inference step.

> **Note**: The general quantization format and kernel expectations align with TensorRT-LLM specifications.

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

#### Visualizing `ColumnMajorTileInterleave<Rows, Cols>`
To answer the specific layout question: **Yes, the tile is effectively stored in Column-Major format.**

For `<64, 2>` (Interleaving 2 columns for every 64 rows):
1.  The logical block is `64 Rows x 2 Columns`.
2.  **Storage Order**:
    *   Store all 64 elements of **Column 0**.
    *   Followed immediately by all 64 elements of **Column 1**.
3.  **Global Structure**: The entire matrix is composed of these `(64x2)` tiles stacked vertically (down K) and then horizontally (across N). Memory is a linear sequence of these tiles.

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

## 7. SwiGLU Details

The operator supports **SwiGLU** activation with support for interleaved inputs, which is critical for performance in certain model architectures (e.g., GPT-OSS).

### 7.1 Formula
The SwiGLU activation is computed as:
```
SwiGLU(x) = Gate * Sigmoid(alpha * Gate) * (Value + beta)
```
Where the input `x` contains both `Gate` and `Value` components.

### 7.2 MoE (Float16/BFloat16) Runtime Fusion
For the standard **MoE** operator (non-quantized), the `Compute` method includes logic to automatically fuse split weight tensors at runtime.

*   **Trigger**: If the optional `fc3_experts_weights` input is provided.
*   **Behavior**:
    *   The operator allocates a temporary buffer.
    *   It manually concatenates `fc1` (Gate) and `fc3` (Value) for each expert.
    *   **Resulting Layout**: `[Expert0: FC1|FC3, Expert1: FC1|FC3, ...]`.
    *   This fused buffer is then passed to the kernel, simulating `swiglu_fusion=2` (Block Fusion).
*   **Activation Check**: This path is taken implicitly when `fc3` is present, typically used with Gated activations like `SiLU` (Mixtral) or `SwiGLU`.

> **Note**: This runtime packing is specific to **standard MoE**. The **QMoE** operator does **not** perform runtime fusion; correct packing must be done offline (see Section 6).

### 7.3 Fusion Modes (`swiglu_fusion`)
The operator handles three distinct modes for SwiGLU, controlled by the `swiglu_fusion` attribute:

1.  **No Fusion (`swiglu_fusion=0`)**:
    *   **Inputs**: 3 distinct weight tensors (`fc1`, `fc2`, `fc3`).
    *   **Logic**: `fc1` (Gate) and `fc3` (Value) are provided separately. The kernel handles the computation and activation conceptually as if they were separate comparisons.

2.  **Interleaved Fusion (`swiglu_fusion=1`)**:
    *   **Inputs**: 2 distinct weight tensors (`fc1`, `fc2`).
        *   `fc1` contains *both* Gate and Value weights fused.
    *   **Memory Layout**: `[Gate_0, Value_0, Gate_1, Value_1, ..., Gate_N, Value_N]`.
    *   **Cutlass Requirements**: The kernel expects adjacent elements in the GEMM output to correspond to Gate and Value.
    *   **Usage**: Recommended for optimal performance on newer architectures as it aligns with interleaved GEMM optimizations.

3.  **Block Fusion (`swiglu_fusion=2`)**:
    *   **Inputs**: 2 distinct weight tensors (`fc1`, `fc2`).
    *   **Memory Layout**: `[Gate_0 ... Gate_N | Value_0 ... Value_N]` (Concatenated).
    *   **Logic**: The kernel processes the first half as Gate and the second half as Value.

**Weight Conversion**:
When exporting a model to ONNX with SwiGLU, the weights for the Gate and Value projections (typically FC1 and Gate_Proj) are often merged. To use **Interleaved** mode, these weights must be interleaved at the output channel dimension during the export/packing phase so that the GEMM output naturally results in `[G, V, G, V...]`.

## 8. Summary of Kernel Changes from TensorRT-LLM

The Cutlass kernels in this implementation are derived from TensorRT-LLM but have been significantly enhanced to support broader ONNX Runtime requirements and fix specific issues.

### Key Modifications:

1.  **Pre-Packed ZP/Bias Optimization**:
    *   Implemented `PrePack` logic to pre-process Zero Points offline (or at initialization).
    *   **8-bit**: Pre-calculates `Bias = (128 - ZP) * Scale` to handle the weight shift efficiently.
    *   **4-bit**: Unpacks and stores unscaled ZPs to match Cutlass kernel requirements.
    *   *Commit*: "prepack update", "fix 8-bit/4-bit asymmetric parity"

2.  **SwiGLU Interleaving**:
    *   Enhanced activation kernels to support **Interleaved** SwiGLU (as described in Section 7), allowing direct compatibility with weights packed for interleaved output.
    *   *Commit*: "fix swiglu test", "swiglu parameters"

3.  **Sparse Mixer Support**:
    *   Added support for Sparse Mixer architectures (controlled by `use_sparse_mixer`).
    *   *Commit*: "add sparse mixer"

## 9. Test Status
As of the latest validation:
*   **All tests in `test_moe_cuda.py` pass**, verifying correctness for FP16, Int8, and 4-bit configurations, including SwiGLU and standard activations.
