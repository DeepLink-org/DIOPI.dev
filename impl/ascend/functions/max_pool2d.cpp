/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cmath>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

void MaxPool2dCheck(diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation) {
    AscendTensor inputTemp(input);
    ASCEND_CHECK_ABORT((kernel_size.len == 1 || kernel_size.len == 2), "max_pool2d: kernel_size must either be a single int, or a tuple of two ints");
    ASCEND_CHECK_ABORT((stride.len == 0 || stride.len == 1 || stride.len == 2),
                       "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
    ASCEND_CHECK_ABORT((padding.len == 1 || padding.len == 2), "max_pool2d: padding must be either be a single int, or a tuple of two ints");
    ASCEND_CHECK_ABORT((dilation.len == 1 || dilation.len == 2), "max_pool2d: dilation must be either a single int, or a tuple of two ints");
    ASCEND_CHECK_ABORT((inputTemp.dim() == 3 || inputTemp.dim() == 4), "non-empty 3D or 4D (batch mode) tensor expected for input")
}

diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input,
                                       diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode);
diopiError_t diopiMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride,
                            diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    diopiTensorHandle_t indices;
    makeTensorLike(ctx, &indices, out, diopi_dtype_uint16);
    return diopiMaxPool2dWithIndices(ctx, out, indices, input, kernel_size, stride, padding, dilation, ceil_mode);

    // MaxPool2dCheck(input, kernel_size, stride, padding, dilation);
    // AscendTensor inputTemp(input);
    // AscendTensor outputTemp(out);

    // if (inputTemp.dim() == 3) {
    //     inputTemp.unsqueeze(0);
    //     outputTemp.unsqueeze(0);
    // }

    // auto format = getAclDataFormat(input);
    // const std::string dataFormat = (format == ACL_FORMAT_NHWC) ? "NHWC" : "NCHW";
    // std::cout << std::endl;
    // std::cout << "dataFormat = " << dataFormat << std::endl;

    // std::vector<int64_t> strideTemp(4, 1);
    // std::vector<int64_t> ksizeTemp(4, 1);
    // std::vector<int64_t> paddingTemp(4, 1);
    // std::vector<int64_t> dilationTemp(4, 1);

    // const int64_t kernelH = kernel_size.data[0];
    // const int64_t kernelW = kernel_size.len == 1 ? kernelH : kernel_size.data[1];

    // int64_t channelH;
    // int64_t channelW;

    // // NHWC
    // if (format == ACL_FORMAT_NHWC) {
    //     channelH = 1;
    //     channelW = 2;
    //     // NCHW
    // } else {
    //     channelH = 2;
    //     channelW = 3;
    // }

    // // stride setting
    // if (stride.len == 0) {
    //     strideTemp[channelH] == kernelH;
    //     strideTemp[channelW] == kernelW;
    // } else {
    //     if (stride.len == 1) {
    //         strideTemp[channelH] = stride.data[0];
    //         strideTemp[channelW] = stride.data[0];
    //     } else {
    //         strideTemp[channelH] = stride.data[0];
    //         strideTemp[channelW] = stride.data[1];
    //     }
    // }

    // // dialtion setting
    // if (dilation.len == 1) {
    //     dilationTemp[channelH] = dilation.data[0];
    //     dilationTemp[channelW] = dilation.data[0];
    // } else if (dilation.len == 2) {
    //     dilationTemp[channelH] = dilation.data[0];
    //     dilationTemp[channelW] = dilation.data[1];
    // }

    // // padding setting
    // if (padding.len == 1) {
    //     paddingTemp[channelH] = padding.data[0];
    //     paddingTemp[channelW] = padding.data[0];
    // } else if (padding.len == 2) {
    //     paddingTemp[channelH] = padding.data[0];
    //     paddingTemp[channelW] = padding.data[1];
    // }

    // // kernel_size setting
    // ksizeTemp[channelH] = kernelH;
    // ksizeTemp[channelW] = kernelW;

    // AclOpRunner<1, 1> runner("MaxPoolV3", ctx);
    // runner.addInput(inputTemp)
    //     .setAttr("ksize", ksizeTemp)
    //     .setAttr("strides", strideTemp)
    //     .setAttr("padding_mode", std::string{"CALCULATED"})
    //     .setAttr("pads", paddingTemp)
    //     .setAttr("data_format", dataFormat)
    //     .setAttr("dilations", dilationTemp)
    //     .setAttr("ceil_mode", ceil_mode)
    //     .addOutput(outputTemp);
    // runner.run();
    // std::vector<int64_t> outputShape = outputTemp.shape();
    // std::cout << std::endl;
    // std::cout << "output_shape = ";
    // for (int64_t i : outputShape) {
    //     std::cout << i << " ";
    // }
    // std::cout << "output_format = " << getAclDataFormat(outputTemp.tensorHandle()) << std::endl;
    // std::cout << std::endl;
    // return diopiSuccess;
}

diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input,
                                       diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    MaxPool2dCheck(input, kernel_size, stride, padding, dilation);
    AscendTensor inputAt(input);
    AscendTensor outputAt(out);
    diopiTensorHandle_t indicesTemp;
    diopiDtype_t indicesDtype;
    diopiGetTensorDtype(indices, &indicesDtype);
    if (diopi_dtype_uint16 != indicesDtype) {
        makeTensorLike(ctx, &indicesTemp, indices, diopi_dtype_uint16);
    } else {
        indicesTemp = indices;
    }
    AscendTensor indicesTempAt(indicesTemp);

    int64_t kernelH, kernelW, strideH, strideW, paddingH, paddingW, dilationH, dilationW;

    if (3 == inputAt.dim()) {
        inputAt.unsqueeze(0);
        outputAt.unsqueeze(0);
        indicesTempAt.unsqueeze(0);
    }
    if (1 == kernel_size.data[0]) {
        kernelH = kernel_size.data[0];
        kernelW = kernelH;
    } else {
        kernelH = kernel_size.data[0];
        kernelW = kernel_size.data[1];
    }
    if (0 == stride.len) {
        strideH = kernelH;
        strideW = kernelW;
    } else if (1 == stride.len) {
        strideH = stride.data[0];
        strideW = strideH;
    } else {
        strideH = stride.data[0];
        strideW = stride.data[1];
    }
    if (0 == padding.len) {
        paddingH = 0;
        paddingW = 0;
    } else if (1 == padding.len) {
        paddingH = padding.data[0];
        paddingW = paddingH;
    } else {
        paddingH = padding.data[0];
        paddingW = padding.data[1];
    }
    if (0 == dilation.len) {
        dilationH = 1;
        dilationW = 1;
    } else if (1 == dilation.len) {
        dilationH = dilation.data[0];
        dilationW = dilationH;
    } else {
        dilationH = dilation.data[0];
        dilationW = dilation.data[1];
    }
    

    // std::vector<int64_t> strideTemp(4, 1);
    // std::vector<int64_t> ksizeTemp(4, 1);
    // std::vector<int64_t> paddingTemp(4, 1);
    // std::vector<int64_t> dilationTemp(4, 1);

    // const int64_t kernelH = kernel_size.data[0];
    // const int64_t kernelW = kernel_size.len == 1 ? 1 : kernel_size.data[1];
    // int64_t channelH;
    // int64_t channelW;

    // auto format = getAclDataFormat(input);
    // const std::string dataFormat = (format == ACL_FORMAT_NHWC) ? "NHWC" : "NCHW";
    // std::cout << std::endl;
    // std::cout << "dataFormat = " << dataFormat << std::endl;
    // // NHWC
    // if (format == ACL_FORMAT_NHWC) {
    //     channelH = 1;
    //     channelW = 2;
    //     // NCHW
    // } else {
    //     channelH = 2;
    //     channelW = 3;
    // }

    // // stride setting
    // if (stride.len == 0) {
    //     strideTemp[channelH] ==G 1;
    //     strideTemp[channelW] == 1;
    // } else {
    //     if (stride.len == 1) {
    //         strideTemp[channelH] = stride.data[0];
    //         strideTemp[channelW] = stride.data[0];
    //     } else {
    //         strideTemp[channelH] = stride.data[0];
    //         strideTemp[channelW] = stride.data[1];
    //     }
    // }

    // // dialtion setting
    // if (dilation.len == 1) {
    //     dilationTemp[channelH] = dilation.data[0];
    //     dilationTemp[channelW] = dilation.data[0];
    // } else if (dilation.len == 2) {
    //     dilationTemp[channelH] = dilation.data[0];
    //     dilationTemp[channelW] = dilation.data[1];
    // }

    // // padding setting
    // if (padding.len == 1) {
    //     paddingTemp[channelH] = padding.data[0];
    //     paddingTemp[channelW] = padding.data[0];
    // } else if (padding.len == 2) {
    //     paddingTemp[channelH] = padding.data[0];
    //     paddingTemp[channelW] = padding.data[1];
    // }

    // // kernel_size setting
    // ksizeTemp[channelH] = kernelH;
    // ksizeTemp[channelW] = kernelW;

    // printf("----- %d\n", indicesTempAt.dtype());
    // printf("----- %d\n", indicesTempAt.getAclDataType());

    int c0 = 16, c1;
    AscendTensor indicesNewAt;
    c1 = 0 == indicesTempAt.shape(1) % c0 ? indicesTempAt.shape(1) / c0 : indicesTempAt.shape(1) / c0 + 1;
    makeTensor(ctx,
               indicesNewAt,
               {
                   indicesTempAt.shape(0),
                   c1,
                   indicesTempAt.shape(2),
                   indicesTempAt.shape(3),
                   c0,
               },
               {
                   c0 * indicesTempAt.shape(3) * indicesTempAt.shape(2) * c1,
                   c0 * indicesTempAt.shape(3) * indicesTempAt.shape(2),
                   c0 * indicesTempAt.shape(3),
                   c0,
                   1,
               },
               diopi_dtype_uint16,
               diopi_device);
    AscendTensor outputNewAt;
    c1 = 0 == outputAt.shape(1) % c0 ? outputAt.shape(1) / c0 : outputAt.shape(1) / c0 + 1;
    makeTensor(ctx,
               outputNewAt,
               {
                   outputAt.shape(0),
                   c1,
                   outputAt.shape(2),
                   outputAt.shape(3),
                   c0,
               },
               {
                   c0 * outputAt.shape(3) * outputAt.shape(2) * c1,
                   c0 * outputAt.shape(3) * outputAt.shape(2),
                   c0 * outputAt.shape(3),
                   c0,
                   1,
               },
               outputAt.dtype(),
               diopi_device);

    printf("###############\n");
    printContiguousTensor(ctx, inputAt, "input");
    printContiguousTensor(ctx, outputAt, "output");
    printContiguousTensor(ctx, indicesNewAt, "indicesNew");
    printf("###############\n");

    AclOpRunner<1, 2> runner("MaxPoolWithArgmaxV1", ctx);
    runner.addInput(inputAt)
        .setAttr("ksize", std::vector<int64_t>{1, kernelH, kernelW, 1})
        .setAttr("strides", std::vector<int64_t>{1, strideH, strideW, 1})
        // .setAttr("padding_mode", std::string{"CALCULATED"})
        .setAttr("pads", std::vector<int64_t>{1, paddingH, paddingW, 1})
        .setAttr("dilations", std::vector<int64_t>{1, dilationH, dilationW, 1})
        .setAttr("ceil_mode", ceil_mode)
        .addOutput(outputNewAt, ACL_FORMAT_NC1HWC0)
        .addOutput(indicesNewAt, ACL_FORMAT_NC1HWC0);
    runner.run();

    std::vector<int64_t> outputShape = outputAt.shape();
    std::cout << std::endl;
    std::cout << "output_shape = ";
    for (int64_t i : outputShape) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    // reshape(ctx, outputTemp, outputTemp, )
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
