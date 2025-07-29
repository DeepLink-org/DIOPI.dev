/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiLinalgQR(diopiContextHandle_t ctx, diopiConstTensorHandle_t A, const char* mode, diopiTensorHandle_t Q, diopiTensorHandle_t R) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnLinalgQr, ctx, A, mode, Q, R);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
 