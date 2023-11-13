/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiThreshold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* threshold,
                            const diopiScalar_t* value) {
    diopiTensorHandle_t thresholdTensor;
    diopiTensorHandle_t valueTensor;
    makeTensorLike(ctx,&valueTensor,input);
    makeTensorLike(ctx,&thresholdTensor,input);
    makeTensorFromScalar(ctx, threshold, &thresholdTensor);
    makeTensorFromScalar(ctx, value, &valueTensor);
    AclOpRunner<1, 1>("ThresholdV2", ctx).addInput(input).setAttr("threshold",threshold->fval).setAttr("value",value->ival).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiThresholdInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* threshold, const diopiScalar_t* value) {
    return diopiThreshold(ctx, input, input, threshold, value);
}

diopiError_t diopiThresholdBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                    diopiConstTensorHandle_t input, const diopiScalar_t* threshold) {
    AclOpRunner<2, 1>("ThresholdGradV2D", ctx).addInput(gradOutput).addInput(input).setAttr("threshold", getValue<float>(threshold)).addOutput(gradInput).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
