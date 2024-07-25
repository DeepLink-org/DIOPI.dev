/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */
#include <c10/util/Backtrace.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <diopi/functions.h>
#include <torch/nn.h>
#include <torch/optim.h>
#include <torch/torch.h>

#include <cstddef>

#include "helper.hpp"

static const char* name = "MuxiDevice";
const char* diopiGetVendorName() { return name; }

namespace impl {
namespace muxi {

diopiError_t diopiCat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t insNum, int64_t dim) {
    impl::aten::ContextManger contextManger(ctx);
    DIOPI_CHECK_PTR(tensors);
    auto tensorWrapperList = impl::aten::buildATenList(tensors, insNum);
    std::vector<at::Tensor> tensorList;
    tensorList.resize(insNum);
    for (size_t i = 0; i < insNum; i++) {
        tensorList[i] = tensorWrapperList[i];
    }
    auto atOut = impl::aten::buildATen(out);
    std::cout << __FUNCTION__ << ":" << __LINE__ << std::endl;
    at::cat_out(atOut, tensorList, dim);
    std::cout << __FUNCTION__ << ":" << __LINE__ << std::endl << "backtrace:" << c10::get_backtrace() << std::endl;
    ;
    return diopiSuccess;
}

}  // namespace muxi
}  // namespace impl
