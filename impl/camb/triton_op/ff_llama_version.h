#ifndef TT_KERNEL_FF_LLAMA_VERSION_INCLUDES
#define TT_KERNEL_FF_LLAMA_VERSION_INCLUDES

#include "cnrt.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// tt-linker: ff_llama_version:void* a_ptr, void* w1_ptr, void* w3_ptr, void* out_ptr, void* rms_w_ptr, int32_t M, int32_t N, int32_t K, int32_t stride_am, int32_t stride_ak, int32_t stride_w1k, int32_t stride_w1n, int32_t stride_w3k, int32_t stride_w3n, int32_t stride_outm, int32_t stride_outn, int32_t stride_rms_w
/*
 Function ff_llama_version.
 enable linear memory: True
*/
cnrtRet_t ff_llama_version(cnrtQueue_t queue, cnrtDim3_t* dim, void* a_ptr, void* w1_ptr, void* w3_ptr, void* out_ptr, void* rms_w_ptr, int32_t M, int32_t N, int32_t K, int32_t stride_am, int32_t stride_ak, int32_t stride_w1k, int32_t stride_w1n, int32_t stride_w3k, int32_t stride_w3n, int32_t stride_outm, int32_t stride_outn, int32_t stride_rms_w);

#ifdef __cplusplus
}
#endif

#endif
