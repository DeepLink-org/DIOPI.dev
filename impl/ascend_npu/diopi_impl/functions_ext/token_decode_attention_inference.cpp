/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <torch/torch.h>

#include "../helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiTokenDecodeAttentionInferenceV1(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t q,
                                                diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                                diopiConstTensorHandle_t b_loc, diopiConstTensorHandle_t b_start_loc, diopiConstTensorHandle_t b_seq_len,
                                                int max_input_len, int other_kv_index) {
    BEGIN_CALL_ACL_OP(out, q, k, v, b_loc, b_start_loc, b_seq_len);
    int batch = qAt.size(0);
    int head_num_q = qAt.size(1);
    int dim = qAt.size(2);
    int hidden_size_q = head_num_q * dim;
    int head_num_kv = kAt.size(1);
    int hidden_size_kv = head_num_kv * dim;
    double scaleValue = 1. / std::sqrt(dim);
    qAt = qAt.reshape({batch, 1, hidden_size_q});
    c10::ScalarType dtype = qAt.scalar_type();
    c10::Device device = qAt.device();
    c10::Layout layout = qAt.layout();
    at::Tensor bSeqLenCpu = b_seq_lenAt.cpu();
    at::Tensor bStartLocCpu = b_start_locAt.cpu();
    c10::optional<at::Tensor> paddingMask = c10::nullopt;
    c10::optional<at::Tensor> attnMask = c10::nullopt;
    c10::IntArrayRef actSeqLen = {};
    for (int i = 0; i < batch; ++i) {
        int curSeqLen = bSeqLenCpu[i].item<int>();
        int curSeqStartLoc = bStartLocCpu[i].item<int>();
        at::Tensor kvLoc = at::index_select(b_locAt[i], 0, acl_op::arange(max_input_len - curSeqLen, max_input_len, at::kInt, layout, device, false));
        at::Tensor key = at::index(kAt, {kvLoc}).view({1, curSeqLen, hidden_size_kv});
        at::Tensor value = at::index(vAt, {kvLoc}).view({1, curSeqLen, hidden_size_kv});
        at::Tensor query = at::slice(qAt, 0, i, i+1, 1);
        auto outAtReshaped = at::slice(outAt, 0, i, i+1, 1).reshape({1, 1, hidden_size_q});
        at::TensorList keyList = key;
        at::TensorList valueList = value;
        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnIncreFlashAttention, query, keyList, valueList, paddingMask, attnMask, actSeqLen,
                                     head_num_q, scaleValue, "BSH", head_num_kv, outAtReshaped);
    }
    END_CALL_ACL_OP();
}

diopiError_t diopiTokenDecodeAttentionInferenceV2(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t q,
                                                diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                                diopiConstTensorHandle_t b_loc, diopiConstTensorHandle_t b_start_loc, diopiConstTensorHandle_t b_seq_len,
                                                int max_input_len, int other_kv_index) {
    BEGIN_CALL_ACL_OP(out, q, k, v, b_loc, b_start_loc, b_seq_len);
    int64_t batch = qAt.size(0);
    int64_t head_num_q = qAt.size(1);
    int64_t dim = qAt.size(2);
    int64_t hidden_size_q = head_num_q * dim;
    int64_t head_num_kv = kAt.size(1);
    int64_t hidden_size_kv = head_num_kv * dim;
    int64_t max_seq_len_in_kv = kAt.size(0) / batch;
    double scaleValue = 1. / std::sqrt(dim);

    c10::optional<at::Tensor> paddingMask = c10::nullopt;
    c10::optional<at::Tensor> attnMask = c10::nullopt;
    std::array<int64_t, 3> q_bsh_array = {batch, 1, hidden_size_q};
    std::array<int64_t, 3> kv_bsh_array = {batch, max_seq_len_in_kv, hidden_size_kv};
    at::Tensor query = qAt.reshape(q_bsh_array);
    kAt = kAt.reshape(kv_bsh_array);
    vAt = vAt.reshape(kv_bsh_array);
    at::TensorList keyList = kAt;
    at::TensorList valueList = vAt;
    at::Tensor outAtReshaped = outAt.reshape(q_bsh_array);
    b_seq_lenAt = b_seq_lenAt.cpu().to(c10::ScalarType::Long);
    at::IntArrayRef b_seq_len_array(b_seq_lenAt.data_ptr<long>(), batch);
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnIncreFlashAttention, query, keyList, valueList, paddingMask, attnMask, b_seq_len_array,
                                 head_num_q, scaleValue, "BSH", head_num_kv, outAtReshaped);
    END_CALL_ACL_OP();
}

diopiError_t diopiTokenDecodeAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t q,
                                                diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                                diopiConstTensorHandle_t b_loc, diopiConstTensorHandle_t b_start_loc, diopiConstTensorHandle_t b_seq_len,
                                                int max_input_len, int other_kv_index) {
#if 0
    return diopiTokenDecodeAttentionInferenceV1(ctx, out, q, k, v, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index);
#else
    return diopiTokenDecodeAttentionInferenceV2(ctx, out, q, k, v, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index);
#endif
}

diopiError_t diopiTokenDecodeAttentionInferenceBatchOne(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t q,
                                                        diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                                        diopiConstTensorHandle_t b_loc, diopiConstTensorHandle_t b_start_loc, diopiConstTensorHandle_t b_seq_len,
                                                        int max_input_len, int other_kv_index) {
    BEGIN_CALL_ACL_OP(out, q, k, v, b_seq_len);
    int head_num_q = qAt.size(1);
    int dim = qAt.size(2);
    int hidden_size_q = head_num_q * dim;
    int head_num_kv = kAt.size(1);
    int hidden_size_kv = head_num_kv * dim;
    double scaleValue = 1. / std::sqrt(dim);
    qAt = qAt.reshape({1, 1, hidden_size_q});
    at::Tensor bSeqLenCpu = b_seq_lenAt.cpu();
    int curSeqLen = bSeqLenCpu[0].item<int>();
    at::Tensor key = at::slice(kAt, 0, 0, curSeqLen, 1).view({1, curSeqLen, hidden_size_kv});
    at::Tensor value = at::slice(vAt, 0, 0, curSeqLen, 1).view({1, curSeqLen, hidden_size_kv});
    auto outAtReshaped = outAt.reshape({1, 1, hidden_size_q});
    c10::optional<at::Tensor> paddingMask = c10::nullopt;
    c10::optional<at::Tensor> attnMask = c10::nullopt;
    c10::IntArrayRef actSeqLen = {};
    at::TensorList keyList = key;
    at::TensorList valueList = value;
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnIncreFlashAttention, qAt, keyList, valueList, paddingMask, attnMask, actSeqLen,
                                 head_num_q, scaleValue, "BSH", head_num_kv, outAtReshaped);
    END_CALL_ACL_OP();
}

diopiError_t diopiIncreFlashAttention(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t q,
                                      diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t paddingMask, diopiConstTensorHandle_t attenMask,
                                      diopiSize_t actualSeqLengths, int64_t numHeads, double scaleValue, const char* inputLayout, int64_t numKeyValueHeads) {
    BEGIN_CALL_ACL_OP(out, q, k, v, paddingMask, attenMask, actualSeqLengths);
    at::Tensor result = op_api::npu_incre_flash_attention(qAt, kAt, vAt, paddingMaskAt, attenMaskAt, actualSeqLengthsAt, numHeads, scaleValue, inputLayout, numKeyValueHeads);
    outAt.copy_(result);
    END_CALL_ACL_OP()
}

}  // namespace OP_IMPL_NS
