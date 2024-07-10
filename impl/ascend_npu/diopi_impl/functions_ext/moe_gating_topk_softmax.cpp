/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../helper.hpp"
#include "ATen/core/TensorBody.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiMoeGatingTopKSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t routing_weights, diopiTensorHandle_t selected_experts,
                                       diopiTensorHandle_t selected_idx, diopiTensorHandle_t router_logits, int64_t topk) {
BEGIN_CALL_ACL_OP(routing_weights, selected_experts, selected_idx, router_logits);
at::Tensor finishedOptional = at::Tensor();
EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnMoeGatingTopKSoftmax,
                             router_logitsAt,
                             finishedOptional,
                             topk,
                             routing_weightsAt,
                             selected_expertsAt,
                             selected_idxAt);
END_CALL_ACL_OP();
}
}
