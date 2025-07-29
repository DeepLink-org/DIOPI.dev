/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
 
diopiError_t diopiRoll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t shifts, diopiSize_t dims) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnRoll, ctx, input, shifts, dims, out);
    return diopiSuccess;
}
 
}  // namespace ascend
}  // namespace impl
 