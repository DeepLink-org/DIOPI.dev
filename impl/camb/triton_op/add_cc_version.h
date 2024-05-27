#ifndef TT_KERNEL_ADD_CC_VERSION_INCLUDES
#define TT_KERNEL_ADD_CC_VERSION_INCLUDES

#include "cnrt.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// tt-linker: add_cc_version:void* x_ptr, void* y_ptr, void* output_ptr, int32_t n_elements
/*
 Function add_cc_version.
 enable linear memory: False
*/
cnrtRet_t add_cc_version(cnrtQueue_t queue, cnrtDim3_t* dim, void* x_ptr, void* y_ptr, void* output_ptr, int32_t n_elements);

#ifdef __cplusplus
}
#endif

#endif
