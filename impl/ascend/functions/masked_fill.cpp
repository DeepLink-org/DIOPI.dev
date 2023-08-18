/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

DIOPI_API diopiError_t diopiMaskedFillInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
    AclOpRunner<3, 1>("MaskedFill", ctx).addInput(input).addInput(mask).addInput(value).addOutput(input).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
