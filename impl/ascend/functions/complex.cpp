/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiComplex(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t real, diopiConstTensorHandle_t imag) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnComplex, ctx, real, imag, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
  