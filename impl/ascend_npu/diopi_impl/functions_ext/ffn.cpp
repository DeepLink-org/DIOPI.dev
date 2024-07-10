/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../helper.hpp"
#include "op_plugin/utils/op_api_common.h"
#include <iostream>

namespace OP_IMPL_NS {

diopiError_t diopiFFN(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t query, diopiTensorHandle_t weight1,
                      diopiTensorHandle_t weight2, diopiSize_t expertTokens, const char* activation) {
    BEGIN_CALL_ACL_OP(out, query, weight1, weight2);
    at::Tensor bias1, bias2, scale, offset, deqScale1, deqScale2, antiquant_scale1, antiquant_scale2, antiquant_offset1, antiquant_offset2;
    int64_t innerPrecise = 1;
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFFN,
                                 queryAt,
                                 weight1At,
                                 weight2At,
                                 expertTokens,
                                 bias1,
                                 bias2,
                                 scale,
                                 offset,
                                 deqScale1,
                                 deqScale2,
                                 antiquant_scale1,
                                 antiquant_scale2,
                                 antiquant_offset1,
                                 antiquant_offset2,
                                 activation,
                                 innerPrecise,
                                 outAt);
    END_CALL_ACL_OP();
}
}  // namespace OP_IMPL_NS
