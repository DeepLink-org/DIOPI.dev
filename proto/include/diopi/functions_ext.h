/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, OpenComputeLab.
 */

#ifndef _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_EXT_H_
#define _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_EXT_H_

#include <diopi/diopirt.h>

#if defined(__cplusplus)
extern "C" {
#endif  // __cplusplus


/**
 * @brief Apply rotary embedding operation to an input tensor.
 * @param[in] ctx Context environment.
 * @param[out] out The output tensor containing the rotary embeddings. type = [float32, float16, float64].
 * @param[in] x The input tensor which rotary embedding will be applied. type = [float32, float16, float64].
 * @param[in] cos The cosine values. type = [float32, float16, float64].
 * @param[in] sin The sine values. type = [float32, float16, float64].
 * @param[in] conj bool: If `false`, computes the complex conjugate of the rotary embeddings for forward. If `true`, computes regular rotary embeddings for
 * backward.
 * @param[in] interleaved bool:
 *   - When set to `false`, rotary embedding is applied by splitting 'x' in half and separately applying sine and cosine to each half.
 *   - When set to `true`, rotary embedding is applied by pairing every two elements in 'x' and applying sine and cosine to each pair.
 */
DIOPI_API diopiError_t diopiRotaryEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t x, diopiConstTensorHandle_t cos,
                                            diopiConstTensorHandle_t sin, const bool conj, const bool interleaved);

/**
 * @brief Apply Root Mean Square (RMS) Normalization to the input tensor.
 * @param[in] ctx Context environment.
 * @param[out] out the output tensor containing the normalized values. type = [float32, float16, float64].
 * @param[in] invRMS The tensor containing the inverse of root mean square. type = [float32, float16, float64].
 * @param[in] input The input tensor to be normalized. type = [float32, float16, float64].
 * @param[in] normalized_shape The shape of the normalization. type = [int32, int64].
 * @param[in] weight The gain parameter used to re-scale the standardized summed inputs type = [float32, float16, float64].
 * @param[in] bias The bias tensor for the normalization. type = [float32, float16, float64].
 * @param[in] eps A small value to avoid division by zero. type = [float64].
 */
DIOPI_API diopiError_t diopiRMSNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t invRMS, diopiConstTensorHandle_t input,
                                    diopiSize_t normalized_shape, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, double eps);

/**
 * @brief Compute the backward pass for Root Mean Square (RMS) Normalization.
 * @param[in] ctx Context environment.
 * @param[out] gradInput The gradient of the input tensor. type = [float32, float16, float64].
 * @param[out] gradWeight The gradient of the weight parameter. type = [float32, float16, float64].
 * @param[out] gradBias The gradient of the bias parameter. type = [float32, float16, float64].
 * @param[in] gradOutput The gradient of the output from the forward pass. type = [float32, float16, float64].
 * @param[in] input The input tensor used in the forward pass. type = [float32, float16, float64].
 * @param[in] weight The weight parameter used in the forward pass. type = [float32, float16, float64].
 * @param[in] bias The bias used in the forward pass. type = [float32, float16, float64].
 * @param[in] invRMS The inverse of the root mean square values computed in the forward pass. type = [float32, float16, float64].
 * @param[in] normalized_shape The shape of the normalization. type = [int32, int64].
 * @param[in] eps A small value used in the computation to avoid division by zero. type = [float64].
 */
DIOPI_API diopiError_t diopiRMSNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight,
                                            diopiTensorHandle_t gradBias, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                            diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiConstTensorHandle_t invRMS,
                                            diopiSize_t normalized_shape, double eps);


/**
 * @brief Compute the forward pass for MultiheadAttention.
 * @param[in] ctx Context environment.
 * @param[in] q Query tensor. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] k Key tensor. shape = [batch_size, k_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] v Value tensor. shape = [batch_size, v_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] dropout_p Dropout probability. type = [float32, float16, float64].
 * @param[in] is_causal Flag to determine if the attention should be causal, masking future tokens. type = [bool]
 * @param[in] return_debug_mask Flag indicating if the attention debug mask should be returned. type = [bool].
 * @param[in] scale Scaling factor for attention weights. type = [float32, float16, float64].
 * @param[out] out Tensor containing the result after applying multi-head attention. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [float32,
 * float16, float64].
 * @param[out] softmax_lse Tensor representing the log-sum-exp of the softmax values. shape = [batch_size, head_num, q_seq_len]. type = [float32, float16,
 * float64].
 * @param[out] gen Handle for the random number generator used in dropout.
 * @param[out] debug_attn_mask Debugging tensor for the attention mask (returned if return_debug_mask is true). shape = [batch_size, num_heads, q_seq_len,
 * k_seq_len]. type = [bool].
 */
DIOPI_API diopiError_t diopiMultiHeadAttention(diopiContextHandle_t ctx, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                               double dropout_p, bool is_causal, bool return_debug_mask, double scale, diopiTensorHandle_t out,
                                               diopiTensorHandle_t softmax_lse, diopiGeneratorHandle_t gen, diopiTensorHandle_t debug_attn_mask);

/**
 * @brief Compute the forward pass for MultiheadAttention.
 * @param[in] ctx Context environment.
 * @param[in] grad_out The gradient of the output tensor. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] q Query tensor from the forward pass. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] k Key tensor from the forward pass. shape = [batch_size, k_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] v Value tensor from the forward pass. shape = [batch_size, v_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] out Output tensor from the forward pass. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] softmax_lse Tensor representing the log-sum-exp of softmax values from the forward pass. shape = [batch_size, head_num, q_seq_len]. type =
 * [float32, float16, float64].
 * @param[in] dropout_p Dropout probability. type = [float32, float16, float64].
 * @param[in] is_causal Flag to determine if the attention should be causal, masking future tokens. type = [bool]
 * @param[in] gen Handle representing the random number generator used for dropout in the forward pass.
 * @param[in] scale Scaling factor used for attention weights in the forward pass. type = [float32, float16, float64].
 * @param[out] grad_q The gradient of the query tensor. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 * @param[out] grad_k The gradient of the key tensor. shape = [batch_size, k_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 * @param[out] grad_v The gradient of the value tensor. shape = [batch_size, v_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 */
DIOPI_API diopiError_t diopiMultiHeadAttentionBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t q,
                                                       diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t out,
                                                       diopiConstTensorHandle_t softmax_lse, double dropout_p, bool is_causal, diopiGeneratorHandle_t gen,
                                                       double scale, diopiTensorHandle_t grad_q, diopiTensorHandle_t grad_k, diopiTensorHandle_t grad_v);

/**
 * @brief Compute the forward pass for MultiheadAttentionVarLen.
 * @param[in] ctx Context environment.
 * @param[in] q Query tensor. shape = [q_nums, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] k Key tensor. shape = [k_nums, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] v Value tensor. shape = [v_nums, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] cum_seq_q Cumulative sequence length for the query. shape = [batch_size+1, ]. type = [int64, int32].
 * @param[in] cum_seq_k Cumulative sequence length for the key. shape = [batch_size+1, ]. type = [int64, int32].
 * @param[in] max_q Maximum sequence length for the query. type = [int64, int32].
 * @param[in] max_k Maximum sequence length for the key. type = [int64, int32].
 * @param[in] dropout_p Dropout probability. type = [float32, float16, float64].
 * @param[in] is_causal Flag to determine if the attention should be causal, masking future tokens. type = [bool]
 * @param[in] return_debug_mask Flag indicating if the attention debug mask should be returned. type = [bool].
 * @param[in] scale Scaling factor for attention weights. type = [float32, float16, float64].
 * @param[out] out Tensor containing the result after applying multi-head attention. shape = [q_nums, head_num, head_dim]. type = [float32, float16, float64].
 * @param[out] softmax_lse Tensor representing the log-sum-exp of the softmax values. shape = [batch_size, head_num, max_q]. type = [float32, float16, float64].
 * @param[out] gen Handle for the random number generator used in dropout.
 * @param[out] debug_attn_mask Debugging tensor for the attention mask (returned if return_debug_mask is true). shape = [batch_size, num_heads, max_q, max_k].
 * type = [bool].
 */
DIOPI_API diopiError_t diopiMultiHeadAttentionVarLen(diopiContextHandle_t ctx, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                                     diopiConstTensorHandle_t v, diopiConstTensorHandle_t cum_seq_q, diopiConstTensorHandle_t cum_seq_k,
                                                     int64_t max_q, int64_t max_k, double dropout_p, bool is_causal, bool return_debug_mask, double scale,
                                                     diopiTensorHandle_t out, diopiTensorHandle_t softmax_lse, diopiGeneratorHandle_t gen,
                                                     diopiTensorHandle_t debug_attn_mask);

/**
 * @brief Compute the forward pass for MultiheadAttentionVarLen.
 * @param[in] ctx Context environment.
 * @param[in] grad_out The gradient of the output tensor. shape = [q_nums, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] q Query tensor from the forward pass. shape = [q_nums, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] k Key tensor from the forward pass. shape = [k_nums, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] v Value tensor from the forward pass. shape = [v_nums, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] out Output tensor from the forward pass. shape = [q_nums, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] softmax_lse Tensor representing the log-sum-exp of softmax values from the forward pass. shape = [batch_size, head_num, max_q]. type = [float32,
 * float16, float64].
 * @param[in] cum_seq_q Cumulative sequence length for the query. shape = [batch_size+1, ]. type = [int64, int32].
 * @param[in] cum_seq_k Cumulative sequence length for the key. shape = [batch_size+1, ]. type = [int64, int32].
 * @param[in] max_q Maximum sequence length for the query. type = [int64, int32].
 * @param[in] max_k Maximum sequence length for the key. type = [int64, int32].
 * @param[in] dropout_p Dropout probability. type = [float32, float16, float64].
 * @param[in] is_causal Flag to determine if the attention should be causal, masking future tokens. type = [bool]
 * @param[in] gen Handle representing the random number generator used for dropout in the forward pass.
 * @param[in] scale Scaling factor used for attention weights in the forward pass. type = [float32, float16, float64].
 * @param[out] grad_q The gradient of the query tensor. shape = [q_nums, head_num, head_dim]. type = [float32, float16, float64].
 * @param[out] grad_k The gradient of the key tensor. shape = [k_nums, head_num, head_dim]. type = [float32, float16, float64].
 * @param[out] grad_v The gradient of the value tensor. shape = [v_nums, head_num, head_dim]. type = [float32, float16, float64].
 */
DIOPI_API diopiError_t diopiMultiHeadAttentionVarLenBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t q,
                                                             diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t out,
                                                             diopiConstTensorHandle_t softmax_lse, diopiConstTensorHandle_t cum_seq_q,
                                                             diopiConstTensorHandle_t cum_seq_k, int64_t max_q, int64_t max_k, double dropout_p, bool is_causal,
                                                             diopiGeneratorHandle_t gen, double scale, diopiTensorHandle_t grad_q, diopiTensorHandle_t grad_k,
                                                             diopiTensorHandle_t grad_v);

/**
 * @brief This function applies a penalty to the given logits based on the presence and frequency of certain tokens in the input sequence to suppress
 * generating tokens repeatedly.
 * The p_cumsum_seq_len is used to determine the sequence length, which is then used to extract the corresponding token_id from p_token_ids and
 * token_count from p_token_counts.
 * For each token，the final logit_value = original_logit_value - corresponding_frequency_penalty * token_count - corresponding_presence_penalty.
 * @param[in] ctx Context environment.
 * @param[inout] Logits Tensor representing the logits. Shape: [batch_size, voc_len]. It contains the predicted scores for each token in the input sequences.
 * It will be penalized by frequency_penalty and presence_penalty.
 * @param[in] presence_penalty Tensor representing the presence penalty for each batch. Shape: [batch_size,]. It contains the penalty values to be subtracted
 * from the logits.
 * @param[in] frequency_penalty Tensor representing the frequency penalty for each batch. Shape: [batch_size,]. It contains the penalty values to be subtracted
 * from the logits.
 * @param[in] p_token_ids Tensor representing the token_ids for generated tokens. Shape:[generated_tokens_num].
 * @param[in] p_token_counts Tensor representing the count of each token for generated tokens. Shape:[generated_tokens_num].
 * @param[in] p_cumsum_seq_len Tensor representing the cumulative sum of sequece lengths of each batch. Shape:[batch_size+1].
 * It contains the indices indicating the satrt and end positions of each batch in the p_token_ids and p_token_counts tensors.
 * @param[in] p_max_len_in_batch: Scalar representing the maximum length among the sequeces in a batch.
 */
DIOPI_API diopiError_t diopiApplyPenalty(diopiContextHandle_t ctx, diopiTensorHandle_t Logits, diopiConstTensorHandle_t presence_penalty,
                                         diopiConstTensorHandle_t frequency_penalty, diopiConstTensorHandle_t p_token_ids,
                                         diopiConstTensorHandle_t p_token_counts, diopiConstTensorHandle_t p_cumsum_seq_len, int p_max_len_in_batch);


#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_EXT_H_

