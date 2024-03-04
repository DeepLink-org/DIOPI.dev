/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

// FIXME(lljbash): This is not working when diopi_adaptor casts inputs' dtype. Check out why.

diopiError_t diopiAddcmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1,
                          diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnAddcmul, ctx, input, tensor1, tensor2, value, out);
#if 0
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    AclOpRunner<4, 1>("Addcmul", ctx).addInput(input).addInput(tensor1).addInput(tensor2).addConstInput(*value, dtype).addOutput(out).run();
    return diopiSuccess;
#endif
}

diopiError_t diopiAddcmulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2,
                             const diopiScalar_t* value) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceAddcmul, ctx, input, tensor1, tensor2, value);
#if 0
    return diopiAddcmul(ctx, input, input, tensor1, tensor2, value);
#endif
}

}  // namespace ascend
}  // namespace impl
