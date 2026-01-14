// Inspired by https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the const bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```

#define BOOL_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                    \
    if (COND) {                            \
      const bool CONST_NAME = true;        \
      return __VA_ARGS__();                \
    } else {                               \
      const bool CONST_NAME = false;       \
      return __VA_ARGS__();                \
    }                                      \
  }()

#define FLASHATTENTION_DISABLE_ALIBI  // TEMP: Remove if we enable alibi
#ifdef FLASHATTENTION_DISABLE_ALIBI
#define ALIBI_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                     \
    const bool CONST_NAME = false;          \
    return __VA_ARGS__();                   \
  }()
#else
#define ALIBI_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
#define EVENK_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                     \
    const bool CONST_NAME = true;           \
    return __VA_ARGS__();                   \
  }()
#else
#define EVENK_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_SOFTCAP
#define SOFTCAP_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                       \
    const bool CONST_NAME = false;            \
    return __VA_ARGS__();                     \
  }()
#else
#define SOFTCAP_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_LOCAL
#define LOCAL_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                     \
    const bool CONST_NAME = false;          \
    return __VA_ARGS__();                   \
  }()
#else
#define LOCAL_SWITCH BOOL_SWITCH
#endif

#define FLASHATTENTION_ASSUME_QUANT_CAUSAL  // GQA assumes causal is true, so we only build causal kernels for quantization.
#ifdef FLASHATTENTION_ASSUME_QUANT_CAUSAL
#define QUANT_CAUSAL_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                            \
    const bool CONST_NAME = true;                  \
    return __VA_ARGS__();                          \
  }()
#else
#define QUANT_CAUSAL_SWITCH BOOL_SWITCH
#endif

// ORT_QUICK_BUILD = 1 only builds fp16 kernels, ORT_QUICK_BUILD = 2 builds both fp16 and bf16 kernels.
#if ORT_QUICK_BUILD == 1
// Quick build mode: only fp16 kernels are compiled
#define FP16_SWITCH(COND, ...)         \
  [&] {                                \
    using elem_type = cutlass::half_t; \
    return __VA_ARGS__();              \
  }()
#else
#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = cutlass::half_t;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = cutlass::bfloat16_t; \
      return __VA_ARGS__();                  \
    }                                        \
  }()
#endif

#ifdef ORT_QUICK_BUILD
// Quick build mode: only hdim128 kernels are compiled
#define HEADDIM_SWITCH(HEADDIM, ...) \
  [&] {                              \
    const int kHeadDim = 128;        \
    return __VA_ARGS__();            \
  }()
#else
#define HEADDIM_SWITCH(HEADDIM, ...) \
  [&] {                              \
    if (HEADDIM <= 32) {             \
      const int kHeadDim = 32;       \
      return __VA_ARGS__();          \
    } else if (HEADDIM <= 64) {      \
      const int kHeadDim = 64;       \
      return __VA_ARGS__();          \
    } else if (HEADDIM <= 96) {      \
      const int kHeadDim = 96;       \
      return __VA_ARGS__();          \
    } else if (HEADDIM <= 128) {     \
      const int kHeadDim = 128;      \
      return __VA_ARGS__();          \
    } else if (HEADDIM <= 192) {     \
      const int kHeadDim = 192;      \
      return __VA_ARGS__();          \
    } else if (HEADDIM <= 256) {     \
      const int kHeadDim = 256;      \
      return __VA_ARGS__();          \
    }                                \
  }()
#endif
