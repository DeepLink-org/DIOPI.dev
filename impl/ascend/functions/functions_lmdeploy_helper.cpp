/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <acl/acl.h>
#include <acl/acl_op.h>
#include <acl/acl_op_compiler.h>
#include <diopi/functions.h>
#include <diopi/functions_lmdeploy.h>
#include <math.h>

#include <cassert>

#include "../common/acloprunner.hpp"
// #include <torch/nn.h>
// #include <torch/optim.h>
// #include <torch/torch.h>
#include <cstring>

#define FLT_MIN __FLT_MIN__
#define FLT_MAX __FLT_MAX__

extern "C" {

#define CALL_ACLRT(Expr)                                                                          \
    do {                                                                                          \
        TRACK_ACL(#Expr);                                                                         \
        ::aclError ret = Expr;                                                                    \
        if (ret != ::ACL_SUCCESS) {                                                               \
            throw std::runtime_error(std::string("ascend device error:") + aclGetRecentErrMsg()); \
        }                                                                                         \
    } while (0);

DIOPI_API diopiError_t diopiLmdeployCopyH2D(diopiContextHandle_t ctx, diopiTensorHandle_t dst, diopiConstTensorHandle_t src, bool async) {
    diopiDevice_t dst_dev;
    diopiGetTensorDevice(dst, &dst_dev);
    diopiDevice_t src_dev;
    diopiGetTensorDevice(src, &src_dev);
    if (dst_dev != diopiDevice_t::diopi_device || src_dev != diopiDevice_t::diopi_host) {
        return diopiErrorOccurred;
    }

    diopiStreamHandle_t stream;
    diopiGetStream(ctx, &stream);
    int64_t numel;
    diopiGetTensorNumel(src, &numel);
    int64_t esize;
    diopiGetTensorElemSize(src, &esize);
    if (nullptr == dst || nullptr == src) {
        return diopiErrorOccurred;
    }
    if (async) {
        CALL_ACLRT(::aclrtMemcpyAsync(dst, numel * esize, src, numel * esize, ACL_MEMCPY_HOST_TO_DEVICE, stream));
    } else {
        CALL_ACLRT(::aclrtMemcpy(dst, numel * esize, src, numel * esize, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLmdeployCopyD2H(diopiContextHandle_t ctx, diopiTensorHandle_t dst, diopiConstTensorHandle_t src, bool async) {
    diopiDevice_t dst_dev;
    diopiGetTensorDevice(dst, &dst_dev);
    diopiDevice_t src_dev;
    diopiGetTensorDevice(src, &src_dev);
    if (dst_dev != diopiDevice_t::diopi_host || src_dev != diopiDevice_t::diopi_device) {
        return diopiErrorOccurred;
    }

    diopiStreamHandle_t stream;
    diopiGetStream(ctx, &stream);
    int64_t numel;
    diopiGetTensorNumel(src, &numel);
    int64_t esize;
    diopiGetTensorElemSize(src, &esize);
    if (nullptr == dst || nullptr == src) {
        return diopiErrorOccurred;
    }
    if (async) {
        CALL_ACLRT(::aclrtMemcpyAsync(dst, numel * esize, src, numel * esize, ACL_MEMCPY_DEVICE_TO_HOST, stream));
    } else {
        CALL_ACLRT(::aclrtMemcpy(dst, numel * esize, src, numel * esize, ACL_MEMCPY_DEVICE_TO_HOST));
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLmdeployCopyD2D(diopiContextHandle_t ctx, diopiTensorHandle_t dst, diopiConstTensorHandle_t src, bool async) {
    diopiDevice_t dst_dev;
    diopiGetTensorDevice(dst, &dst_dev);
    diopiDevice_t src_dev;
    diopiGetTensorDevice(src, &src_dev);
    if (dst_dev != diopiDevice_t::diopi_device || src_dev != diopiDevice_t::diopi_device) {
        return diopiErrorOccurred;
    }

    diopiStreamHandle_t stream;
    diopiGetStream(ctx, &stream);
    int64_t numel;
    diopiGetTensorNumel(src, &numel);
    int64_t esize;
    diopiGetTensorElemSize(src, &esize);
    if (nullptr == dst || nullptr == src) {
        return diopiErrorOccurred;
    }
    if (async) {
        CALL_ACLRT(::aclrtMemcpyAsync(dst, numel * esize, src, numel * esize, ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
    } else {
        CALL_ACLRT(::aclrtMemcpy(dst, numel * esize, src, numel * esize, ACL_MEMCPY_DEVICE_TO_DEVICE));
    }
    return diopiSuccess;
}

}  // extern "C"
