#ifndef TT_KERNEL_SOFTMAX_CC_VERSION_INCLUDES
#define TT_KERNEL_SOFTMAX_CC_VERSION_INCLUDES

#include "cnrt.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// tt-linker: softmax_cc_version:void* output_ptr, void* input_ptr, int32_t input_row_stride, int32_t output_row_stride, int32_t n_cols
/*
 Function softmax_cc_version.
 enable linear memory: True
*/
cnrtRet_t softmax_cc_version(cnrtQueue_t queue, cnrtDim3_t* dim, void* output_ptr, void* input_ptr, int32_t input_row_stride, int32_t output_row_stride, int32_t n_cols);

#ifdef __cplusplus
}
#endif

#endif
