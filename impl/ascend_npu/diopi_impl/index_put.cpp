/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <ATen/native/IndexingUtils.h>
#include <ATen/native/TypeProperties.h>

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/third_party/acl/inc/op_proto/all_ops.h"
#include "op_plugin/utils/AdvancedIndex.h"
#include "op_plugin/utils/OpAdapter.h"

namespace {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;
using npu_compile_type = at_npu::native::CompileType;

template <typename ge_op_type>
at_npu::native::DynamicInputRegFunc indexput_func = [](std::vector<std::pair<uint32_t, uint32_t>> num_and_index, std::string op_name) -> ge::OperatorPtr {
    auto ge_op = std::make_shared<ge_op_type>(op_name.c_str());
    ge_op->create_dynamic_input_byindex_indices(num_and_index.front().first, num_and_index.front().second);
    return ge_op;
};
const std::string x_str = "x";
const std::string value_str = "value";
const std::string indexed_sizes_str = "indexed_sizes";
const std::string indexed_strides_str = "indexed_strides";
const std::string aicore_str = "AiCore";

bool is_aicpu_valid(const at::Tensor& self, const std::vector<at::Tensor>& all_defined_indices, const at::SmallVector<int64_t, N> masks) {
    // using aicpu at non-binary scene
    if (!at_npu::native::env::CheckJitDisable()) {
        return true;
    }
    // using aicore when index is continous, otherwise aicpu
    bool is_zero_in_masks = false;
    for (int32_t i = 0; i < masks.size(); i++) {
        if (is_zero_in_masks && masks[i] == 1) {
            return true;
        }
        if (masks[i] == 0) {
            is_zero_in_masks = true;
        }
    }
    // using aicpu when indices num is more than 20000 or the type of self tensor is double.
    if (self.scalar_type() == at::kDouble || all_defined_indices[0].numel() > 20000) {
        return true;
    }

    // indices may need broadcast, in this case, indexput is implemented by aicpu
    for (int32_t i = 1; i < all_defined_indices.size(); i++) {
        if (all_defined_indices[0].dim() != all_defined_indices[i].dim()) {
            return true;
        }
        for (int32_t j = 0; j < all_defined_indices[0].dim(); j++) {
            if (all_defined_indices[0].sizes()[j] != all_defined_indices[i].sizes()[j]) {
                return true;
            }
        }
    }

    int tail_size = 1;
    for (int32_t i = all_defined_indices.size(); i < self.dim(); i++) {
        tail_size = tail_size * self.sizes()[i];
    }
    if (self.scalar_type() != at::kHalf && self.scalar_type() != at::kFloat && (all_defined_indices[0].numel() > 200 || tail_size > 128)) {
        return true;
    }
    return false;
}

at::Tensor& index_put_aicore_nocheck(at::Tensor& self, const std::vector<at::Tensor>& all_defined_indices, at::SmallVector<int64_t, N> masks,
                                     at::SmallVector<int64_t, N> expand_masks, const at::Tensor& value, bool accumulate) {
    if (value.numel() == 0) {
        return self;
    }
    at::Tensor temp_self = self;
    at::Tensor temp_value = value;
    if (self.scalar_type() == at::ScalarType::Half) {
        temp_self = at_npu::native::custom_ops::npu_dtype_cast(self, at::ScalarType::Float);
        temp_value = at_npu::native::custom_ops::npu_dtype_cast(value, at::ScalarType::Float);
    }
    at::Tensor temp_value_broadcast = temp_value;
    if (self.dim() == 1 && all_defined_indices.size() == 1 && all_defined_indices[0].scalar_type() == at::kLong &&
        all_defined_indices[0].sizes()[0] != value.sizes()[0]) {
        temp_value_broadcast = acl_op::npu_broadcast(temp_value, all_defined_indices[0].sizes());
    }

    at_npu::native::OpCommand cmd;
    cmd.Name("IndexPutV2")
        .Input(temp_self, x_str)
        .Input(temp_value_broadcast, value_str)
        .Input(masks, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT, "", indexed_sizes_str)
        .Input(expand_masks, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT, "", indexed_strides_str);
    for (int i = 0; i < all_defined_indices.size(); i++) {
        string input_name = "indices" + std::to_string(i);
        cmd.Input(all_defined_indices[i], input_name);
    }
    cmd.DynamicInputReg(indexput_func<ge::op::IndexPutV2>, {{all_defined_indices.size(), 4}}).Output(temp_self, x_str).Attr("accumulate", accumulate).Run();
    if (self.scalar_type() == at::ScalarType::Half) {
        temp_self = at_npu::native::custom_ops::npu_dtype_cast(temp_self, at::ScalarType::Half);
        self.copy_(temp_self);
    } else {
        self = temp_self;
    }
    return self;
}

at::SmallVector<int64_t, N> npu_expand_tensors_mask(const at::Tensor& self, const torch::List<c10::optional<at::Tensor>>& indices) {
    at::SmallVector<int64_t, N> result;
    for (c10::optional<at::Tensor> index_opt : indices) {
        if (!index_opt.has_value()) {
            result.emplace_back(0);
        } else {
            const auto& index = *index_opt;
            if (index.scalar_type() != at::kByte && index.scalar_type() != at::kBool) {
                result.emplace_back(0);
                break;
            }
        }
    }
    if (result.empty()) {
        result.emplace_back(1);
    }
    return result;
}

at::Tensor& index_put_aicpu_nocheck(at::Tensor& result, const at::Tensor& self, std::vector<at::Tensor> all_defined_indices, at::SmallVector<int64_t, N> masks,
                                    const at::Tensor& value, bool accumulate) {
    if (value.numel() == 0) {
        return result;
    }

    at::Tensor temp_self = self;
    at::Tensor temp_value = value;
    if (self.scalar_type() == at::ScalarType::Half) {
        temp_self = at_npu::native::custom_ops::npu_dtype_cast(self, at::ScalarType::Float);
        temp_value = at_npu::native::custom_ops::npu_dtype_cast(value, at::ScalarType::Float);
        result = at_npu::native::custom_ops::npu_dtype_cast(result, at::ScalarType::Float);
    }

    at_npu::native::OpCommand cmd;
    cmd.Name("IndexPutV2")
        .Input(temp_self, x_str)
        .Input(temp_value, value_str)
        .Input(masks, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT, "", indexed_sizes_str)
        .Input(masks, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT, "", indexed_strides_str);
    for (int i = 0; i < all_defined_indices.size(); i++) {
        string input_name = "indices" + std::to_string(i);
        cmd.Input(all_defined_indices[i], input_name);
    }
    cmd.DynamicInputReg(indexput_func<ge::op::IndexPutV2>, {{all_defined_indices.size(), 4}})
        .Output(result, x_str)
        .Attr("_exclude_engines", aicore_str)
        .Attr("accumulate", accumulate)
        .Run();

    if (self.scalar_type() == at::ScalarType::Half) {
        result = at_npu::native::custom_ops::npu_dtype_cast(result, at::ScalarType::Half);
    }
    return result;
}

at::Tensor& index_put_aicpu(at::Tensor& result, at::Tensor& self, std::vector<at::Tensor> all_defined_indices, at::SmallVector<int64_t, N> masks,
                            const at::Tensor& value, bool accumulate) {
    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        index_put_aicpu_nocheck(contiguous_self, contiguous_self, all_defined_indices, masks, value, accumulate);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        index_put_aicpu_nocheck(self, self, all_defined_indices, masks, value, accumulate);
    }
    return self;
}

at::Tensor& index_put_aicore(at::Tensor& self, std::vector<at::Tensor> indices_expand, at::SmallVector<int64_t, N> masks,
                             at::SmallVector<int64_t, N> bool_masks, const at::Tensor& value, bool accumulate) {
    // value broadcast
    auto index_output_size = op_infer::index_npu_output_size(self, indices_expand);
    auto value_shape = op_infer::array_to_small_vector(value.sizes());
    at::Tensor value_broadcast = (index_output_size != value_shape) ? acl_op::npu_broadcast(value, index_output_size) : value;

    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        index_put_aicore_nocheck(contiguous_self, indices_expand, masks, bool_masks, value_broadcast, accumulate);
        self.copy_(contiguous_self);
    } else {
        index_put_aicore_nocheck(self, indices_expand, masks, bool_masks, value_broadcast, accumulate);
    }
    return self;
}

at::Tensor& indexPutInp(at::Tensor& self, const c10::List<c10::optional<at::Tensor>>& indices, const at::Tensor& value, const bool accumulate,
                        const bool unsafe) {
    if (self.device().type() == at::kCPU) {
        return at::native::_index_put_impl_(self, indices, value, accumulate, unsafe);
    }
    at::native::checkIndexTensorTypes(indices);
    at::SmallVector<int64_t, N> masks;
    std::vector<at::Tensor> all_defined_indices;
    std::vector<at::Tensor> indices_expand;
    c10::List<c10::optional<at::Tensor>> indices_expand_list;
    indices_expand = op_plugin::AdvanceIndex::npu_expand_tensors(self, indices);
    for (at::Tensor index_opt : indices_expand) {
        indices_expand_list.push_back(index_opt);
    }
    auto info = op_plugin::AdvanceIndex::make_info(self, indices_expand_list);
    TORCH_CHECK(op_plugin::AdvanceIndex::is_expandable_to(value.sizes(), info.src.sizes()),
                "shape mismatch: value tensor of shape ",
                value.sizes(),
                " cannot be broadcast to indexing result of shape ",
                info.src.sizes());
    for (c10::optional<at::Tensor> index_opt : indices_expand) {
        if (index_opt.has_value()) {
            const auto& index = *index_opt;
            if (index.defined()) {
                all_defined_indices.emplace_back(index);
                masks.emplace_back(1);
            } else {
                masks.emplace_back(0);
            }
        } else {
            masks.emplace_back(0);
        }
    }
    for (auto& all_defined_indice : all_defined_indices) {
        if (all_defined_indice.device() != self.device()) {
            all_defined_indice = all_defined_indice.to(self.device());
        }
    }

    npu_preparation::CastBackToOriFormat(self);
    at::Tensor value_copy = value;
    at::Tensor self_copy = self;
    npu_preparation::CastBackToOriFormat(value_copy);

    bool aicpu_true = is_aicpu_valid(self, all_defined_indices, masks);
    if (aicpu_true) {
        index_put_aicpu(self_copy, self_copy, all_defined_indices, masks, value_copy, accumulate);
    } else {
        auto bool_mask = npu_expand_tensors_mask(self, indices);
        index_put_aicore(self_copy, indices_expand, masks, bool_mask, value_copy, accumulate);
    }
    self.copy_(self_copy);
    return self;
}

}  // namespace

namespace OP_IMPL_NS {

diopiError_t diopiIndexPut(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t values,
                           diopiConstTensorHandle_t* indices, int64_t indicesCounts, bool accumulate) {
    BEGIN_CALL_ACL_OP(out, input, values);
    DIOPI_CHECK_PTR(indices);
    // handle empty tensor
    if (outAt.numel() == 0) {
        return diopiSuccess;
    }
    outAt.copy_(inputAt);
    c10::List<c10::optional<at::Tensor>> indicesAtList;
    indicesAtList.reserve(indicesCounts);
    assert(indicesCounts >= 1);
    for (int i = 0; i < indicesCounts; ++i) {
        indicesAtList.emplace_back(impl::aten::buildATen(indices[i]));
    }
    indexPutInp(outAt, indicesAtList, valuesAt, accumulate, false);
    END_CALL_ACL_OP();
}

diopiError_t diopiIndexPutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t values, diopiConstTensorHandle_t* indices,
                              int64_t indicesCounts, bool accumulate) {
    BEGIN_CALL_ACL_OP(input, values);
    DIOPI_CHECK_PTR(indices);
    // handle empty tensor
    if (inputAt.numel() == 0) {
        return diopiSuccess;
    }
    c10::List<c10::optional<at::Tensor>> indicesAtList;
    indicesAtList.reserve(indicesCounts);
    assert(indicesCounts >= 1);
    for (int i = 0; i < indicesCounts; ++i) {
        indicesAtList.emplace_back(impl::aten::buildATen(indices[i]));
    }
    indexPutInp(inputAt, indicesAtList, valuesAt, accumulate, false);
    END_CALL_ACL_OP();
}
}  // namespace OP_IMPL_NS