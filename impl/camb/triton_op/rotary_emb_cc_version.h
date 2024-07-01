#ifndef TT_KERNEL_ROTARY_EMB_CC_VERSION_INCLUDES
#define TT_KERNEL_ROTARY_EMB_CC_VERSION_INCLUDES

#include "cnrt.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// tt-linker: rotary_emb_cc_version:void* OUT, void* X, void* COS, void* SIN, int32_t SEQLEN_OFFSETS, int32_t seqlen, int32_t nheads, int32_t rotary_dim, int32_t seqlen_ro, int32_t stride_out_batch, int32_t stride_out_seqlen, int32_t stride_out_nheads, int32_t stride_out_headdim, int32_t stride_x_batch, int32_t stride_x_seqlen, int32_t stride_x_nheads, int32_t stride_x_headdim
/*
 Function rotary_emb_cc_version.
 enable linear memory: True
*/
cnrtRet_t rotary_emb_cc_version(cnrtQueue_t queue, cnrtDim3_t* dim, void* OUT, void* X, void* COS, void* SIN, int32_t SEQLEN_OFFSETS, int32_t seqlen, int32_t nheads, int32_t rotary_dim, int32_t seqlen_ro, int32_t stride_out_batch, int32_t stride_out_seqlen, int32_t stride_out_nheads, int32_t stride_out_headdim, int32_t stride_x_batch, int32_t stride_x_seqlen, int32_t stride_x_nheads, int32_t stride_x_headdim);

#ifdef __cplusplus
}
#endif

#endif
