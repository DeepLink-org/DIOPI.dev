/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiSin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnSin, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiSinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceSin, ctx, input);
    return diopiSuccess;
}

diopiError_t diopiAsin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnAsin, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiAsinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceAsin, ctx, input);
    return diopiSuccess;
}

diopiError_t diopiSinh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnSinh, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiSinhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceSinh, ctx, input);
    return diopiSuccess;
}

diopiError_t diopiAsinh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnAsinh, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiAsinhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input){
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceAsinh, ctx, input);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
