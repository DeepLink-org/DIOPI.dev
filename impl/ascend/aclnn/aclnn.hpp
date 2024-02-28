#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attention_score.h"
#include "aclnnop/aclnn_flash_attention_score_grad.h"
#include "impl_functions.hpp"

namespace impl {
namespace ascend {

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)
#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int aclnnFlashAttentionAdaptor(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiTensorHandle_t* softmaxMax, diopiTensorHandle_t* softmaxSum,
                               diopiTensorHandle_t* softmaxOut, diopiGeneratorHandle_t gen, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                               diopiConstTensorHandle_t v, double pDropout, double softmaxScale, bool isCausal);

int aclnnFlashAttentionBackwardAdaptor(diopiContextHandle_t ctx, diopiTensorHandle_t gradQ, diopiTensorHandle_t gradK, diopiTensorHandle_t gradV,
                                       diopiConstTensorHandle_t gradOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                       diopiConstTensorHandle_t attentionOut, diopiConstTensorHandle_t softmaxMax, diopiConstTensorHandle_t softmaxSum,
                                       diopiConstTensorHandle_t softmaxOut, diopiGeneratorHandle_t gen, double pDropout, double softmaxScale, bool isCausal);
}  // namespace ascend
}  // namespace impl