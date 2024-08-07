/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"
namespace impl {
namespace ascend {

diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t outputSize) {
    for (int64_t i = 0; i < outputSize.len; i++) {
        if (outputSize.data[i] == 0) {
            return diopiSuccess;
        }
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnAdaptiveAvgPool2d, ctx, input, outputSize, out);
    return diopiSuccess;
}

diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                            diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnAdaptiveAvgPool2dBackward, ctx, gradOutput, input, gradInput);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
