/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiMultiHeadAttnForward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t dropoutMask, diopiTensorHandle_t queryRes,
                                       diopiTensorHandle_t keyRes, diopiTensorHandle_t valueRes, diopiTensorHandle_t attnScores, diopiTensorHandle_t attnRes,
                                       diopiTensorHandle_t queryContext, diopiConstTensorHandle_t query, diopiConstTensorHandle_t key,
                                       diopiConstTensorHandle_t value, diopiConstTensorHandle_t queryWeight, diopiConstTensorHandle_t keyWeight,
                                       diopiConstTensorHandle_t valueWeight, diopiConstTensorHandle_t attnMask, diopiConstTensorHandle_t outProjWeight,
                                       diopiConstTensorHandle_t queryBias, diopiConstTensorHandle_t keyBias, diopiConstTensorHandle_t valueBias,
                                       diopiConstTensorHandle_t outProjBias, diopiConstTensorHandle_t dropoutMaskInput, const int64_t attnHeadNum,
                                       const int64_t attnHeadDim, const int64_t srcLen, const int64_t tgtLen, const double dropoutProb,
                                       const bool softmaxUseFloat) {
    BEGIN_CALL_ACL_OP(out, dropoutMask, queryRes, keyRes, valueRes, attnScores, attnRes, queryContext);
    BEGIN_CALL_ACL_OP(query, key, value, queryWeight, keyWeight, valueWeight, attnMask, outProjWeight);
    BEGIN_CALL_ACL_OP(queryBias, keyBias, valueBias, outProjBias, dropoutMaskInput);

    auto result = acl_op::npu_multi_head_attention(queryAt,
                                                   keyAt,
                                                   valueAt,
                                                   queryWeightAt,
                                                   keyWeightAt,
                                                   valueWeightAt,
                                                   attnMaskAt,
                                                   outProjWeightAt,
                                                   queryBiasAt,
                                                   keyBiasAt,
                                                   valueBiasAt,
                                                   outProjBiasAt,
                                                   dropoutMaskInputAt,
                                                   attnHeadNum,
                                                   attnHeadDim,
                                                   srcLen,
                                                   tgtLen,
                                                   dropoutProb,
                                                   softmaxUseFloat);

    outAt.copy_(std::get<0>(result));
    dropoutMaskAt.copy_(std::get<1>(result));
    queryResAt.copy_(std::get<2>(result));
    keyResAt.copy_(std::get<3>(result));
    valueResAt.copy_(std::get<4>(result));
    attnScoresAt.copy_(std::get<5>(result));
    attnResAt.copy_(std::get<6>(result));
    queryContextAt.copy_(std::get<7>(result));
    END_CALL_ACL_OP();
}

diopiError_t diopiMultiHeadAttnBackward(diopiContextHandle_t ctx, diopiTensorHandle_t queryWeightGrad, diopiTensorHandle_t keyWeightGrad,
                                        diopiTensorHandle_t valueWeightGrad, diopiTensorHandle_t outProjWeightGrad, diopiTensorHandle_t queryGrad,
                                        diopiTensorHandle_t keyGrad, diopiTensorHandle_t valueGrad, diopiTensorHandle_t queryBiasGrad,
                                        diopiTensorHandle_t keyBiasGrad, diopiTensorHandle_t valueBiasGrad, diopiTensorHandle_t outProjBiasGrad,
                                        diopiConstTensorHandle_t query, diopiConstTensorHandle_t key, diopiConstTensorHandle_t value,
                                        diopiConstTensorHandle_t queryWeight, diopiConstTensorHandle_t keyWeight, diopiConstTensorHandle_t valueWeight,
                                        diopiConstTensorHandle_t outProjWeight, diopiConstTensorHandle_t queryBias, diopiConstTensorHandle_t keyBias,
                                        diopiConstTensorHandle_t valueBias, diopiConstTensorHandle_t outProjBias, diopiConstTensorHandle_t queryRes,
                                        diopiConstTensorHandle_t keyRes, diopiConstTensorHandle_t valueRes, diopiConstTensorHandle_t attnScores,
                                        diopiConstTensorHandle_t attnRes, diopiConstTensorHandle_t queryContext, diopiConstTensorHandle_t outGrad,
                                        diopiConstTensorHandle_t dropoutMask, const int64_t attnHeadNum, const int64_t attnHeadDim, const int64_t srcLen,
                                        const int64_t tgtLen, const double dropoutProb, const bool softmaxUseFloat) {
    BEGIN_CALL_ACL_OP(queryWeightGrad, keyWeightGrad, valueWeightGrad, outProjWeightGrad, queryGrad, keyGrad, valueGrad);
    BEGIN_CALL_ACL_OP(queryBiasGrad, keyBiasGrad, valueBiasGrad, outProjBiasGrad, query, key, value, queryWeight);
    BEGIN_CALL_ACL_OP(keyWeight, valueWeight, outProjWeight, queryBias, keyBias, valueBias, outProjBias);
    BEGIN_CALL_ACL_OP(queryRes, keyRes, valueRes, attnScores, attnRes, queryContext, outGrad, dropoutMask);

    auto result = acl_op::npu_multi_head_attention_backward(queryAt,
                                                            keyAt,
                                                            valueAt,
                                                            queryWeightAt,
                                                            keyWeightAt,
                                                            valueWeightAt,
                                                            outProjWeightAt,
                                                            queryBiasAt,
                                                            keyBiasAt,
                                                            valueBiasAt,
                                                            outProjBiasAt,
                                                            queryResAt,
                                                            keyResAt,
                                                            valueResAt,
                                                            attnScoresAt,
                                                            attnResAt,
                                                            queryContextAt,
                                                            outGradAt,
                                                            dropoutMaskAt,
                                                            attnHeadNum,
                                                            attnHeadDim,
                                                            srcLen,
                                                            tgtLen,
                                                            dropoutProb,
                                                            softmaxUseFloat);
    queryWeightGradAt.copy_(std::get<0>(result));
    keyWeightGradAt.copy_(std::get<1>(result));
    valueWeightGradAt.copy_(std::get<2>(result));
    outProjWeightGradAt.copy_(std::get<3>(result));
    queryGradAt.copy_(std::get<4>(result));
    keyGradAt.copy_(std::get<5>(result));
    valueGradAt.copy_(std::get<6>(result));
    queryBiasGradAt.copy_(std::get<7>(result));
    keyBiasGradAt.copy_(std::get<8>(result));
    valueBiasGradAt.copy_(std::get<9>(result));
    outProjBiasGradAt.copy_(std::get<10>(result));

    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
