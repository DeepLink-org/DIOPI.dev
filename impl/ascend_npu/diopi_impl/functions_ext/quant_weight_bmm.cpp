/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../helper.hpp"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiQuantWeightBatchMatmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t x,
                                  diopiConstTensorHandle_t weight, diopiConstTensorHandle_t antiquantScale,
                                  diopiConstTensorHandle_t antiquantOffsetOptional, diopiConstTensorHandle_t quantScaleOptional,
                                  diopiConstTensorHandle_t quantOffsetOption, diopiConstTensorHandle_t biasOptional,
                                  const int64_t antiquantGroupSize) {
    BEGIN_CALL_ACL_OP(out, x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional, quantOffsetOption, biasOptional);
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnQuantWeightBatchMatmul, xAt, weightAt, antiquantScaleAt, antiquantOffsetOptionalAt, quantScaleOptionalAt, quantOffsetOptionAt, biasOptionalAt, antiquantGroupSize, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
