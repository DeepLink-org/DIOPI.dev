#ifndef TT_KERNEL_RMSNORM_CC_VERSION_INCLUDES
#define TT_KERNEL_RMSNORM_CC_VERSION_INCLUDES

#include "cnrt.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// tt-linker: rmsnorm_cc_version:void* x_ptr, void* rms_w_ptr, void* output_ptr, int32_t stride_x_batch, int32_t stride_x_m, int32_t stride_x_k, int32_t stride_rms_w, int32_t stride_out_batch, int32_t stride_out_m, int32_t stride_out_k
/*
 Function rmsnorm_cc_version.
 enable linear memory: True
*/
cnrtRet_t rmsnorm_cc_version(cnrtQueue_t queue, cnrtDim3_t* dim, void* x_ptr, void* rms_w_ptr, void* output_ptr, int32_t stride_x_batch, int32_t stride_x_m, int32_t stride_x_k, int32_t stride_rms_w, int32_t stride_out_batch, int32_t stride_out_m, int32_t stride_out_k);

#ifdef __cplusplus
}
#endif

#endif
