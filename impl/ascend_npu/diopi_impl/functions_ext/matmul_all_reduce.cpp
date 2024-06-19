/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <ATen/core/TensorBody.h>

#include "../helper.hpp"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiMatmulAllReduce(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t x1, diopiConstTensorHandle_t x2,
                                  diopiConstTensorHandle_t bias, const char* group, const char* reduceOp, int64_t commTurn, int64_t streamMode) {
    BEGIN_CALL_ACL_OP(out, x1, x2, bias);
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnMatmulAllReduce, x1At, x2At, biasAt, group, reduceOp, commTurn, streamMode, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiMatmulAllReduceAddRmsNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t out_bf_norm, diopiConstTensorHandle_t x1,
                                            diopiConstTensorHandle_t x2, diopiConstTensorHandle_t residual, diopiConstTensorHandle_t bias,
                                            diopiConstTensorHandle_t gamma, double eps, const char* group, const char* reduceOp, int64_t commTurn,
                                            int64_t streamMode) {
    BEGIN_CALL_ACL_OP(out, x1, x2, gamma, residual, bias);

    if (residualAt.dim() == 2) {
        residualAt = impl::aten::viewStorage(residualAt, {residualAt.size(0), (int64_t)1, residualAt.size(1)});
        outAt = impl::aten::viewStorage(outAt, {outAt.size(0), (int64_t)1, outAt.size(1)});
    }

    EXEC_NPU_NO_FORMAT_CHECK_CMD(
        aclnnMatmulAllReduceAddRmsNorm, x1At, x2At, biasAt, residualAt, gammaAt, eps, group, reduceOp, commTurn, streamMode, residualAt, outAt);

    if (residualAt.dim() == 3) {
        residualAt = impl::aten::viewStorage(residualAt, {residualAt.size(0), residualAt.size(2)});
        outAt = impl::aten::viewStorage(outAt, {outAt.size(0), outAt.size(2)});
    }

    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
