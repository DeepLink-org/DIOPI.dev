#ifndef TT_KERNEL_ROTARY_EMB_CC_FP16_INCLUDES
#define TT_KERNEL_ROTARY_EMB_CC_FP16_INCLUDES

#include "cnrt.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// tt-linker: rotary_emb_cc_fp16:void* OUT, void* X, void* COS, void* SIN, int32_t seqlen, int32_t nheads, int32_t rotary_dim, int32_t seqlen_ro, int32_t stride_out_batch, int32_t stride_out_seqlen, int32_t stride_out_nheads, int32_t stride_out_headdim, int32_t stride_x_batch, int32_t stride_x_seqlen, int32_t stride_x_nheads, int32_t stride_x_headdim, int32_t CONJUGATE
/*
 Function rotary_emb_cc_fp16.
 enable linear memory: True
*/
cnrtRet_t rotary_emb_cc_fp16(cnrtQueue_t queue, cnrtDim3_t* dim, void* OUT, void* X, void* COS, void* SIN, int32_t seqlen, int32_t nheads, int32_t rotary_dim, int32_t seqlen_ro, int32_t stride_out_batch, int32_t stride_out_seqlen, int32_t stride_out_nheads, int32_t stride_out_headdim, int32_t stride_x_batch, int32_t stride_x_seqlen, int32_t stride_x_nheads, int32_t stride_x_headdim, int32_t CONJUGATE);

#ifdef __cplusplus
}
#endif

#endif
