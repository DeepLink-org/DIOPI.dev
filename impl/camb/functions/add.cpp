/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
#include "../triton_op/add_cc_version.h"
namespace impl {
namespace camb {

diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                      const diopiScalar_t* alpha) {
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor(other);
    DiopiTensor outputTensor(out);
    bool isTriton = ture;
    if(isTriton){
        auto queue = getStream(ctx);
        cnrtDim3_t kDim;
        int clusterCount = 0;
        int corePerCluster = 0;
        cnrtRet_t ret = cnrtDeviceGetAttribute(&clusterCount, cnrtAttrClusterCount, 0);
        if (ret != cnrtSuccess) {
            return diopiErrorOccurred;
        }
        ret = cnrtDeviceGetAttribute(&corePerCluster, cnrtAttrMcorePerCluster, 0);
        if (ret != cnrtSuccess) {
            return diopiErrorOccurred;
        }
        kDim.x = corePerCluster;
        kDim.y = clusterCount;
        kDim.z = 1;
        add_cc_version(queue,&kDim,inputTensor.data(),otherTensor.data(),outputTensor.data(),(int)inputTensor.numel());

    }else{
        DIOPI_CALL(cnnlOpTensor(
        ctx, inputTensor, otherTensor, outputTensor, CNNL_OP_TENSOR_ADD, 1.0, DiopiDataType::isFloatPoint(alpha->stype) ? alpha->fval : alpha->ival));
    }
   
    return diopiSuccess;
}

diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor(other);
    DiopiTensor outputTensor(input);

    DIOPI_CALL(cnnlOpTensor(
        ctx, inputTensor, otherTensor, outputTensor, CNNL_OP_TENSOR_ADD, 1.0, DiopiDataType::isFloatPoint(alpha->stype) ? alpha->fval : alpha->ival));
    return diopiSuccess;
}

diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                            const diopiScalar_t* alpha) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);
    if (inputTensor.dtype() == diopi_dtype_float16 || inputTensor.dtype() == diopi_dtype_float32 ||
        (inputTensor.dtype() == diopi_dtype_int32 && DiopiDataType::isInteger(other->stype) && DiopiDataType::isInteger(alpha->stype))) {
        DIOPI_CALL(cnnlTransformAdaptor(ctx,
                                        outputTensor,
                                        inputTensor,
                                        DiopiDataType::isFloatPoint(other->stype) ? other->fval : other->ival,
                                        DiopiDataType::isFloatPoint(alpha->stype) ? alpha->fval : alpha->ival,
                                        DiopiDataType::isFloatPoint(inputTensor.dtype()) ? 1.0 : 1));
    } else {
        DiopiTensor otherTensor;
        DIOPI_CALL(makeTensorFromScalar(ctx, other, otherTensor));
        DIOPI_CALL(cnnlOpTensor(
            ctx, inputTensor, otherTensor, outputTensor, CNNL_OP_TENSOR_ADD, 1.0, DiopiDataType::isFloatPoint(alpha->stype) ? alpha->fval : alpha->ival));
    }
    return diopiSuccess;
}

diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(input);
    if (inputTensor.dtype() == diopi_dtype_float16 || inputTensor.dtype() == diopi_dtype_float32 ||
        (inputTensor.dtype() == diopi_dtype_int32 && DiopiDataType::isInteger(other->stype) && DiopiDataType::isInteger(alpha->stype))) {
        DIOPI_CALL(cnnlTransformAdaptor(ctx,
                                        outputTensor,
                                        inputTensor,
                                        DiopiDataType::isFloatPoint(other->stype) ? other->fval : other->ival,
                                        DiopiDataType::isFloatPoint(alpha->stype) ? alpha->fval : alpha->ival,
                                        DiopiDataType::isFloatPoint(inputTensor.dtype()) ? 1.0 : 1));
    } else {
        DiopiTensor otherTensor;
        DIOPI_CALL(makeTensorFromScalar(ctx, other, otherTensor));
        DIOPI_CALL(cnnlOpTensor(
            ctx, inputTensor, otherTensor, outputTensor, CNNL_OP_TENSOR_ADD, 1.0, DiopiDataType::isFloatPoint(alpha->stype) ? alpha->fval : alpha->ival));
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
