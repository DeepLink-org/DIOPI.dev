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
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    float InputThreshold = getValue<float>(threshold);
    float InputValue = getValue<float>(value);

    if (dtype == diopi_dtype_uint8) {
        InputThreshold = static_cast<float>(static_cast<uint8_t>(InputThreshold));
        InputValue = static_cast<float>(static_cast<uint8_t>(InputValue));
    }
    if (dtype == diopi_dtype_int32) {
        InputThreshold = static_cast<float>(getValue<int>(threshold));
        InputValue = static_cast<float>(getValue<int>(value));
    }
    AclOpRunner<1, 1>("ThresholdV2D", ctx).addInput(input).setAttr("threshold", InputThreshold).setAttr("value", InputValue).addOutput(out).run();
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
