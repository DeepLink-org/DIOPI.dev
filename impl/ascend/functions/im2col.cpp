/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiIm2Col(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size,
                        diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnIm2col, ctx, input, kernel_size, dilation, padding, stride, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
 