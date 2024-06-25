#ifndef TT_KERNEL_MATMUL_CC_VERSION_INCLUDES
#define TT_KERNEL_MATMUL_CC_VERSION_INCLUDES

#include "cnrt.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// tt-linker: matmul_cc_version:void* a_ptr, void* b_ptr, void* c_ptr, int32_t M, int32_t N, int32_t K, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn, int32_t stride_cm, int32_t stride_cn
/*
 Function matmul_cc_version.
 enable linear memory: True
*/
cnrtRet_t matmul_cc_version(cnrtQueue_t queue, cnrtDim3_t* dim, void* a_ptr, void* b_ptr, void* c_ptr, int32_t M, int32_t N, int32_t K, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn, int32_t stride_cm, int32_t stride_cn);

#ifdef __cplusplus
}
#endif

#endif
