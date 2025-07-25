/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiCosInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCos, ctx, input);
    return diopiSuccess;
}

diopiError_t diopiCos(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnCos, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiAcos(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnAcos, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiAcosInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceAcos, ctx, input);
    return diopiSuccess;
}

diopiError_t diopiCosh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnCosh, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiCoshInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCosh, ctx, input);
    return diopiSuccess;
}

diopiError_t diopiAcosh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnAcosh, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiAcoshInp(diopiContextHandle_t ctx, diopiTensorHandle_t input){
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceAcosh, ctx, input);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
