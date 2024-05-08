/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <c10/util/Optional.h>
#include <torch/torch.h>

#include "../helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"


namespace OP_IMPL_NS {
namespace {
using npu_preparation = at_npu::native::OpPreparation;
}

diopiError_t diopiPagedAttention(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t q,
                                                diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t paddingMask, diopiConstTensorHandle_t attenMask,
                                                diopiSize_t actualSeqLengths, diopiConstTensorHandle_t antiquant_scale, diopiConstTensorHandle_t antiquant_offset, 
                                                diopiConstTensorHandle_t block_table,
                                                diopiConstTensorHandle_t dequant_scale1, diopiConstTensorHandle_t quant_scale1,
                                                diopiConstTensorHandle_t dequant_scale2, diopiConstTensorHandle_t quant_scale2,
                                                diopiConstTensorHandle_t quant_offset2, diopiConstTensorHandle_t kv_padding_size, 
                                                int64_t numHeads, double scaleValue, const char* inputLayout, int64_t numKeyValueHeads,
                                                int64_t block_size, int64_t inner_precise) {
    BEGIN_CALL_ACL_OP(out, q, k, v, paddingMask, attenMask, block_table);
    at::IntArrayRef actSeqLen(actualSeqLengths.data, actualSeqLengths.len);

    // construct the output tensor of the NPU
    at::Tensor output ;
    if (ConvertType(quant_scale2) != nullptr) {
        output = npu_preparation::apply_tensor_without_format(qAt.sizes(), c10::dtype(c10::ScalarType::Char));
    } else if (qAt.dtype() == at::kChar) {
        output = npu_preparation::apply_tensor_without_format(qAt.sizes(), c10::dtype(c10::ScalarType::Half));
    } else {
        output = npu_preparation::apply_tensor_without_format(qAt);
    }

    // convert str
    std::string input_layout_str = std::string(inputLayout);
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    at::TensorList keyTensors = kAt;
    at::TensorList valueTensors = vAt;

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnIncreFlashAttentionV4, qAt, keyTensors, valueTensors, paddingMaskAt, attenMaskAt, actSeqLen,
        dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset,
        block_tableAt,
        kv_padding_size,
        numHeads, scaleValue, input_layout_ptr, numKeyValueHeads, block_size, inner_precise, output);
    
    outAt.copy_(output);
    END_CALL_ACL_OP()
}

}  // namespace OP_IMPL_NS