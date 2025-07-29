/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiMax(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t maxIndices, diopiConstTensorHandle_t input, int64_t dim) {
    AscendTensor inAt(input);
    AscendTensor maxAt(max);
    bool keepdim = false;
    if (inAt.dim() == maxAt.dim()) {
        keepdim = true;
    }
    DIOPI_ASCEND_CALL_ACLNN(aclnnMaxDim, ctx, input, dim, keepdim, max, maxIndices);
    return diopiSuccess;
}

diopiError_t diopiMaxAll(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnMax, ctx, input, max);
    return diopiSuccess;
}

diopiError_t diopiMin(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiTensorHandle_t minIndices, diopiConstTensorHandle_t input, int64_t dim) {
    AscendTensor inAt(input);
    AscendTensor minAt(min);
    bool keepdim = false;
    if (inAt.dim() == minAt.dim()) {
        keepdim = true;
    }
    DIOPI_ASCEND_CALL_ACLNN(aclnnMinDim, ctx, input, dim, keepdim, min, minIndices);
    return diopiSuccess;
}

diopiError_t diopiMinAll(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnMin, ctx, input, min);
    return diopiSuccess;
}

diopiError_t diopiAmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t self, diopiSize_t dim, bool keepdim) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnAmax, ctx, self, dim, keepdim, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
