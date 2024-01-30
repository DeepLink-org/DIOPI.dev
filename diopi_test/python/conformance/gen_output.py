import pickle
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
import math
import torchvision

from gen_input import GenPolicy
from conformance.utils import logger, get_data_from_file
from conformance.db_operation import db_conn
from conformance.exception import GenDataFailedException


def _torch_context_attention(xq, xk, xv, bs, seqlen, num_head, head_dim):
    xq = xq.view(bs, seqlen, num_head, head_dim)
    xk = xk.view(bs, seqlen, num_head, head_dim)
    xv = xv.view(bs, seqlen, num_head, head_dim)
    mask = torch.tril(torch.ones(seqlen, seqlen), diagonal=0).unsqueeze(0).unsqueeze(0).cuda()
    mask[mask == 0.] = -100000000.0
    mask = mask.repeat(bs, num_head, 1, 1)
    keys = xk
    values = xv
    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)
    scores = F.softmax(scores.float() + mask, dim=-1).type_as(xq)
    output = torch.matmul(scores, values).transpose(1, 2).contiguous().reshape(-1, num_head, head_dim)
    return output


class CustomizedTest(object):
    def cast_dtype(input, out):
        out = input.to(out.dtype, copy=True)
        return out

    def meshgrid(tensors, shape=None):
        return torch.meshgrid(tensors)

    def slice_op(input, dim, index):
        sizeI = input.size()
        slice_args = []
        for i in range(len(sizeI)):
            slice_args.append(slice(0, sizeI[i], 1))
        slice_args[dim] = index
        return torch.Tensor.__getitem__(input, slice_args)

    def index(input, **kwargs):
        new_args = []
        for ele in kwargs.values():
            if ele is None:
                hasEllipsis = True
                if hasEllipsis and Ellipsis not in new_args:
                    new_args.append(...)
            else:
                new_args.append(ele)
        return torch.Tensor.__getitem__(input, new_args)

    def sgd(param, param_grad, lr, buf=None, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        param.requires_grad = True
        param.grad = param_grad
        optimizer = torch.optim.SGD([param, ], lr, momentum, dampening, weight_decay, nesterov)
        optimizer.state[param]['momentum_buffer'] = buf
        optimizer.step()
        return param, buf

    def adam(param, param_grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step, amsgrad):
        params_with_grad = [param]
        grads = [param_grad]
        exp_avgs = [exp_avg]
        exp_avg_sqs = [exp_avg_sq]
        max_exp_avg_sqs = [max_exp_avg_sq]
        state_steps = [torch.tensor(float(step))]

        torch.optim._functional.adam(params_with_grad,
                                     grads,
                                     exp_avgs,
                                     exp_avg_sqs,
                                     max_exp_avg_sqs,
                                     state_steps,
                                     amsgrad=amsgrad,
                                     beta1=beta1,
                                     beta2=beta2,
                                     lr=lr,
                                     weight_decay=weight_decay,
                                     eps=eps,
                                     maximize=False)
        return param, param_grad, exp_avg, exp_avg_sq, max_exp_avg_sq

    def adamw(param, param_grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1, beta2, eps, step, weight_decay, amsgrad):
        params_with_grad = [param]
        grads = [param_grad]
        exp_avgs = [exp_avg]
        exp_avg_sqs = [exp_avg_sq]
        max_exp_avg_sqs = [max_exp_avg_sq]
        state_steps = [torch.tensor(float(step))]

        torch.optim._functional.adamw(params_with_grad,
                                      grads,
                                      exp_avgs,
                                      exp_avg_sqs,
                                      max_exp_avg_sqs,
                                      state_steps,
                                      amsgrad=amsgrad,
                                      beta1=beta1,
                                      beta2=beta2,
                                      lr=lr,
                                      weight_decay=weight_decay,
                                      eps=eps,
                                      maximize=False)
        return param, param_grad, exp_avg, exp_avg_sq, max_exp_avg_sq

    def adadelta(param, param_grad, square_avg, acc_delta, lr, rho, eps, weight_decay):
        params_with_grad = [param]
        grads = [param_grad]
        square_avgs = [square_avg]
        acc_deltas = [acc_delta]

        torch.optim._functional.adadelta(params_with_grad,
                                         grads,
                                         square_avgs,
                                         acc_deltas,
                                         lr=lr,
                                         rho=rho,
                                         eps=eps,
                                         weight_decay=weight_decay,
                                         maximize=False)
        return param, param_grad, square_avg, acc_delta

    def rmsprop(param, param_grad, square_avg, grad_avg, momentum_buffer, lr, alpha, eps, weight_decay, momentum, centered):
        params = [param]
        grads = [param_grad]
        square_avgs = [square_avg]
        grad_avgs = [grad_avg]
        momentum_buffer_list = [momentum_buffer]

        torch.optim._functional.rmsprop(params,
                                        grads,
                                        square_avgs,
                                        grad_avgs,
                                        momentum_buffer_list,
                                        lr=lr,
                                        alpha=alpha,
                                        eps=eps,
                                        weight_decay=weight_decay,
                                        momentum=momentum,
                                        centered=centered)
        return param, param_grad, square_avg, grad_avg, momentum_buffer

    def index_put(input, values, indices1, indices2=None, indices3=None, accumulate=False):
        indices = [indices1]
        if indices2 is not None:
            indices.append(indices2)
        if indices3 is not None:
            indices.append(indices3)
        return torch.index_put(input, indices, values, accumulate)

    def im2col(input, kernel_size, dilation=1, padding=0, stride=1):
        return torch.nn.Unfold(kernel_size, dilation, padding, stride)(input)

    def col2im(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
        return torch.nn.Fold(output_size, kernel_size, dilation, padding, stride)(input)

    def clip_grad_norm_(tensors, max_norm, norm_type=2.0, error_if_nonfinite=False):
        parameters = []
        if torch.is_tensor(tensors):
            tensors = [tensors]
        for grad in tensors:
            tensor = torch.empty_like(grad)
            tensor.grad = grad
            parameters.append(tensor)
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite)

    def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False):
        log_probs_ = log_probs.log_softmax(2)
        loss = torch.nn.functional.ctc_loss(log_probs_, targets, input_lengths, target_lengths, blank=blank, reduction=reduction, zero_infinity=zero_infinity)
        return loss

    def linalgqr(input, mode):
        q, r = torch.linalg.qr(input, mode)
        out = [q, r]
        return out

    def batch_norm_stats(input, eps):
        mean, invstd = torch.batch_norm_stats(input, eps)
        out = (mean, invstd)
        return out

    def batch_norm_gather_stats_with_counts(input, mean_all, invstd_all, running_mean, running_var, momentum, eps, count_all):
        mean, invstd = torch.batch_norm_gather_stats_with_counts(input, mean_all, invstd_all, running_mean, running_var, momentum, eps, count_all)
        out = (mean, invstd)
        return out

    def batch_norm_backward_reduce(grad_output, input, mean, invstd, weight, input_g, weight_g, bias_g):
        sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(grad_output, input, mean, invstd, weight, input_g, weight_g, bias_g)
        out = (sum_dy, sum_dy_xmu, grad_weight, grad_bias)
        return out

    def batch_norm_backward_elemt(grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count):
        grad_input = torch.batch_norm_backward_elemt(grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count)
        out = grad_input
        return out

    def batch_norm_elemt(input, weight, bias, mean, invstd, eps):
        out = torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
        return out

    def rotary_emb(input, cos, sin, conj):
        x1, x2 = input.chunk(2, dim=-1)
        data_type = input.dtype
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)
        if not conj:
            out1 = x1 * cos - x2 * sin
            out2 = x1 * sin + x2 * cos
        else:
            out1 = x1 * cos + x2 * sin
            out2 = -x1 * sin + x2 * cos
        out1 = out1.to(data_type)
        out2 = out2.to(data_type)
        out = torch.cat((out1, out2), dim=-1)
        return out

    def rms_norm(input, normalized_shape, weight, bias, eps):
        variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + eps)
        out = weight * input
        return out

    def multihead_attention_forward(q, k, v, dropout_p, is_causal, return_debug_mask, scale):
        # 为了保证精度，因此在test的时候不使用dropout
        from einops import rearrange
        import math

        _, seqlen = q.shape[0], q.shape[1]
        softmax_scale = 1.0 / math.sqrt(q.shape[-1]) if not scale else scale
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        if is_causal:
            causal_mask = torch.triu(
                torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
            )
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        output = torch.einsum("bhts,bshd->bthd", attention, v)
        return output

    def apply_penalty(logits, presence_penalty, frequency_penalty, p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch):
        batch = logits.shape[0]
        for i in range(batch):
            cur_batch_start_index = p_cumsum_seq_len[i]
            cur_batch_end_index = p_cumsum_seq_len[i + 1]
            cur_logits = logits[i, p_token_ids[cur_batch_start_index:cur_batch_end_index]]
            cur_logits = cur_logits - p_token_counts[cur_batch_start_index:cur_batch_end_index] * frequency_penalty[i] - presence_penalty[i]
            logits[i, p_token_ids[cur_batch_start_index:cur_batch_end_index]] = cur_logits
        return logits

    def destindex_copy_kv(k, dest_loc, out):
        out[dest_loc] = k
        return out

    def token_attention(q, k, out, b_loc, b_start_loc, b_seq_len, max_input_len):
        batch, head, dim = b_loc.shape[0], q.shape[1], q.shape[2]
        q_device = q.device
        xq = q.view(batch, 1, head, dim).transpose(1, 2)
        for i in range(batch):
            k_loc = b_loc[i][max_input_len - b_seq_len[i] + torch.arange(0, b_seq_len[i], device=q_device)]
            key = k[k_loc, :].view(1, b_seq_len[i], head, dim).transpose(1, 2)
            out_loc = b_start_loc[i] + torch.arange(0, b_seq_len[i], device=q_device)
            out[:, out_loc] = (torch.matmul(xq[i, :], key.transpose(2, 3)) / math.sqrt(dim)).reshape(head, b_seq_len[i])
        return out

    def token_softmax_reducev(logics, v, out, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index):
        batch, head, dim = b_loc.shape[0], v.shape[1], v.shape[2]
        for i in range(batch):
            v_loc = b_loc[i][max_input_len - b_seq_len[i] + torch.arange(0, b_seq_len[i], device=logics.device)]
            P = logics[:, b_start_loc[i]:b_start_loc[i] + b_seq_len[i]].softmax(-1).reshape(head, 1, 1, b_seq_len[i]).transpose(0, 1)
            V = v[v_loc, :].view(1, b_seq_len[i], head, dim).transpose(1, 2)
            out[i, :] = torch.matmul(P, V).view(1, head, dim)
        return out

    def context_attention(q, k, v, out, b_start_loc, b_seq_len, max_input_len):
        batch, head, dim = b_start_loc.shape[0], q.shape[1], q.shape[2]
        for i in range(batch):
            start = b_start_loc[i]
            end = start + b_seq_len[i]
            out[start:end, :] = _torch_context_attention(q[start:end], k[start:end], v[start:end], 1, int(b_seq_len[i]), head, dim)
        return out

    def plus_scalar_inp(inoutput, val, size):
        if size >= inoutput.shape[0]:
            inoutput += val
        else:
            inoutput[:size] += val
        return inoutput

    def update_padding_count(total_padding_count, input_lengths, max_input_length, batch_size):
        for i in range(batch_size):
            total_padding_count[i] = max_input_length - input_lengths[i]
        return total_padding_count

    def length_criterion(finished, should_stop, finished_sum, sequence_limit_length, batch_size, step):
        finished = sequence_limit_length <= step
        finished_sum = finished.sum()
        should_stop = finished_sum == batch_size
        return finished, should_stop, finished_sum
    
    def gather_output(output_ids, ids, context_length, max_context_len, max_gen_step, max_output_len, batch_size):
        for i in range(batch_size):
            output_ids[i, 0: context_length[i].item()] = ids[0:context_length[i].item(), i]
            if max_gen_step > max_context_len:
                output_ids[i, context_length[i].item():context_length[i].item() + max_gen_step - max_context_len] = ids[max_context_len: max_gen_step, i]
        return output_ids
    
    def banbadwords_inp(logits, output_ids, bad_words, id_offset, bad_words_len, share_words, batch_size, vocab_size, step):
        for i in range(batch_size):
            bad_words_array = bad_words[0] if share_words else bad_words[i][0]
            bad_words_offsets = bad_words[1] if share_words else bad_words[i][1]
            for bad_word_idx in range(bad_words_len):
                if bad_words_offsets[bad_word_idx] < 0:
                    continue
                
                bad_word_start_idx = bad_words_offsets[bad_word_idx - 1].item() if bad_word_idx > 0 else 0
                bad_word_end_idx = bad_words_offsets[bad_word_idx].item()
                bad_word_len = bad_word_end_idx - bad_word_start_idx
                bad_word = bad_words_array[bad_word_start_idx: bad_word_end_idx]
                
                if step + 1 < bad_word_len or bad_word_len < 1:
                    continue
                
                should_ban = bad_word_len == 1
                if bad_word_len != 1:
                    output_ids_to_compare = output_ids[step - (bad_word_len - 1): step, i]
                    bad_word_to_compare = bad_word[:-1]
                    should_ban = (bad_word_to_compare == output_ids_to_compare).all()
                
                if should_ban:
                    banned_token = bad_word[-1].item()
                    if 0 < banned_token and banned_token < vocab_size:
                        logits[i, banned_token] = -float('inf')
        return logits
    
    def stopwords_criterion(output_ids, stop_words, finished, id_offset, stop_words_len, batch_size, step):
        for i in range(batch_size):
            stop_words_array = stop_words[i][0]
            stop_words_offsets = stop_words[i][1]
            for stop_word_idx in range(stop_words_len):
                if stop_words_offsets[stop_word_idx] < 0:
                    continue
                
                stop_word_start_idx = stop_words_offsets[stop_word_idx - 1].item() if stop_word_idx > 0 else 0
                stop_word_end_idx = stop_words_offsets[stop_word_idx].item()
                stop_word_len = stop_word_end_idx - stop_word_start_idx
                stop_word = stop_words_array[stop_word_start_idx: stop_word_end_idx]
                
                if step + 1 < stop_word_len:
                    continue
                
                output_ids_to_compare = output_ids[step + 1 - stop_word_len: step + 1, i]
                should_stop = (stop_word == output_ids_to_compare).all()
                
                if should_stop:
                    finished[i] = True
                    break
        return finished

    def embedding_lookup_pos_encoding(from_tensor, embedding_table, all_ids, batch_size, hidden_units, step):
        this_step_ids = all_ids[step]
        from_tensor = torch.index_select(embedding_table, 0, this_step_ids)
        return from_tensor
    
    def inputids_embedding_lookup_pos_encoding(from_tensor, input_ids, embedding_table, input_lengths, hidden_units):
        from_tensor = torch.index_select(embedding_table, 0, input_ids)
        return from_tensor
    
    def batch_apply_temperature_penalty(logits, bias, temperatures, batch_size, vocab_size, vocab_size_padd):
        inv_temperatures = 1.0 / (temperatures + 1e-6)
        max_t_val = np.finfo(np.float32).max if logits.dtype == torch.float32 else 65504.0
        max_t_val = torch.tensor(max_t_val, dtype=logits.dtype, device=logits.device)
        logits[:, vocab_size: vocab_size_padd] = -max_t_val
        if bias is not None:
            logits[:, :vocab_size] += bias
        logits[:, :vocab_size] *= inv_temperatures.view(-1, 1)
        return logits

    def batch_apply_repetition_penalty(logits, penalties, output_ids, batch_size, vocab_size, input_lengths, max_input_length, step, penalty_type):
        if penalty_type == 0:
            return logits
        
        for i in range(batch_size):
            input_length = input_lengths[i].item() if input_lengths is not None else max_input_length
            index = output_ids[0:input_length, i].view(-1)
            if step >= max_input_length:
                index = torch.cat((index, output_ids[max_input_length:, i].view(-1)))
            logits_this_batch = torch.index_select(logits[i], 0, index)
            # Penalty type.0 == None;1 == Additive means logit - penalty;2 == Multiplicative
            if penalty_type == 1:
                penalty_logits = logits_this_batch - penalties[i]
            else:
                #  penalty_logits = penalty_logits < 0.0f ? logit * penalty[i] : penalty_logits / penalty[i];
                penalty_logits = torch.where(logits_this_batch < 0.0, logits_this_batch * penalties[i], logits_this_batch / penalties[i])
            
            index_int64 = index.to(torch.int64)
            logits[i].scatter_(0, index_int64, penalty_logits)
        
        return logits
    
    def fused_context_attention_inp(inoutput, qkv_weight, qkv_bias, key_cache, value_cache, batch_size, input_lengths, history_lengths, context_lengths, layer_id, local_head_num, local_kv_head_num, size_per_head, max_seq_len, max_q_len, max_kv_len, rotary_embedding, rope_theta):
        token_num = inoutput.shape[0]
        qkv = torch.matmul(inoutput, qkv_weight) # [token_num, hidden_units] * [hidden_units, (local_head_num+local_kv_head_num*2)*size_per_head]
        if qkv_bias is not None:
            qkv += qkv_bias # [token_num, (local_head_num+local_kv_head_num*2)*size_per_head] + [1, local_head_num+local_kv_head_num*2)*size_per_head]
        q = qkv[:, :local_head_num*size_per_head].reshape(token_num, local_head_num, size_per_head)
        k = qkv[:, local_head_num*size_per_head: (local_head_num+local_kv_head_num)*size_per_head].reshape(token_num, local_kv_head_num, size_per_head)
        v = qkv[:, (local_head_num+local_kv_head_num)*size_per_head:].reshape(token_num, local_kv_head_num, size_per_head)
        
        def rope(x: torch.Tensor, rotary_embedding: float, rope_theta: float, t_start: float) -> torch.Tensor:
            # x (torch.Tensor): [input_len, head_num, size_per_head]
            input_len = x.shape[0]
            head_num = x.shape[1]
            x = x.reshape(input_len, head_num, size_per_head // 2, 2)
            # x_odd, x_even: [input_len, head_num, size_per_head // 2, 1]
            x_even = x[:, :, :, 0].reshape(input_len, head_num, size_per_head // 2, 1)
            x_odd = x[:, :, : ,1].reshape(input_len, head_num, size_per_head // 2, 1)
            # theta: [size_per_head // 2], theta[i] = pow(rope_theta, -2 * i / rotary_embedding)
            theta = torch.pow(rope_theta, -2 * torch.arange(size_per_head // 2, dtype=torch.float32, device=x.device) / rotary_embedding)
            # timestamp: [input_len], timestamp[i] = t_start + i
            # sin, cos: [input_len, 1, size_per_head // 2, 1]
            timestamp = torch.arange(x_even.shape[0], dtype=torch.float32, device=x.device) + t_start
            cos = torch.cos(timestamp.view(-1, 1) * theta.view(1, -1)).reshape(input_len, 1, size_per_head // 2, 1)
            sin = torch.sin(timestamp.view(-1, 1) * theta.view(1, -1)).reshape(input_len, 1, size_per_head // 2, 1)
            x_even_new = x_even * cos - x_odd * sin
            x_odd_new = x_odd * cos + x_even * sin
            x_new = torch.cat([x_even_new, x_odd_new], dim=-1).reshape(x.shape[0], x.shape[1], size_per_head)
            x_new = x_new.to(x.dtype)
            return x_new
        
        def get_mask(input_lengths: torch.Tensor, context_lengths: torch.Tensor, max_q_len: int, max_kv_len: int, mask_dtype):
            batch_size = input_lengths.shape[0]
            mask = torch.ones((batch_size, 1, max_q_len, max_kv_len), dtype=mask_dtype, device=input_lengths.device) * -10000.0
            for batch_idx in range(batch_size):
                input_length = input_lengths[batch_idx].item()
                context_length = context_lengths[batch_idx].item()
                mask_this_batch = mask[batch_idx][0] # [max_q_len, max_kv_len] 
                mask_this_batch[:input_length, :context_length - input_length] = 0 
                mask_this_batch[:input_length, context_length - input_length:context_length] = torch.tril(torch.zeros(input_length, input_length, dtype=mask_dtype), diagonal=0) + torch.triu(torch.ones(input_length, input_length, dtype=mask_dtype), diagonal=1) * -10000.0
                
            return mask
        
        pre_batches_len = 0
        # prepare for calculating attention:
        # 1. do rope for q, k
        # 2. store k, v to key_cache, value_cache
        # 3. padding q, k, v for calculating attention
        q_cal = torch.zeros((batch_size, local_head_num, max_q_len, size_per_head), dtype=q.dtype, device=q.device)
        k_cal = torch.zeros((batch_size, local_kv_head_num, size_per_head, max_kv_len), dtype=k.dtype, device=k.device)
        v_cal = torch.zeros((batch_size, local_kv_head_num, max_kv_len, size_per_head), dtype=v.dtype, device=v.device)
        # key_cache & value_cache: [batch_size] of [num_layer, local_kv_head_num, max_seq_len, size_per_head]
        for batch_idx in range(batch_size):
            input_length = input_lengths[batch_idx].item()
            history_length = history_lengths[batch_idx].item()
            # q_this_batch: [input_length, head_num, size_per_head]
            q_this_batch = q[pre_batches_len: pre_batches_len + input_length, :, :]
            k_this_batch = k[pre_batches_len: pre_batches_len + input_length, :, :]
            v_this_batch = v[pre_batches_len: pre_batches_len + input_length, :, :]
            pre_batches_len += input_length
            # RoPE
            q_rope = rope(q_this_batch, rotary_embedding, rope_theta, history_length)
            k_rope = rope(k_this_batch, rotary_embedding, rope_theta, history_length)
            
            # q
            q_cal[batch_idx, :, :input_length, :] = q_rope.permute(1, 0, 2) # [input_length, head_num, size_per_head] -> [head_num, input_length, size_per_head]
            
            # store new k to key_cache
            padd_kv = torch.zeros((max_seq_len - input_length - history_length, local_kv_head_num, size_per_head), dtype=k.dtype, device=k.device)
            k_rope_padd = torch.cat((k_rope, padd_kv), dim=0)
            key_cache[batch_idx][layer_id, :, history_length:, :] = k_rope_padd.permute(1, 0, 2) # [len, head, dim] -> [head, len, dim]
            # load his_k
            his_k = key_cache[batch_idx][layer_id, :, :history_length, :] # [head_num, history_length, size_per_head]
            # k for cal
            k_cal[batch_idx, :, :, :history_length] = his_k.permute(0, 2, 1) # [head, len, dim] -> [head, dim, len]
            k_cal[batch_idx, :, :, history_length:history_length + input_length] = k_rope.permute(1, 2, 0) # [len, head, dim] -> [head, dim, len]
            
                
            # store new v to value_cache
            v_padd = torch.cat((v_this_batch, padd_kv), dim=0)
            value_cache[batch_idx][layer_id, :, history_length:, :] = v_padd.permute(1, 0, 2) # [len, head, dim] -> [head, len, dim]
            # load his_v
            his_v = value_cache[batch_idx][layer_id, :, :history_length, :] # [head_num, history_length, size_per_head]
            # v for cal
            v_cal[batch_idx, :, :history_length, :] = his_v
            v_cal[batch_idx, :, history_length:history_length + input_length, :] = v_this_batch.permute(1, 0, 2) # [len, head, dim] -> [head, len, dim]
        
        # calculate attention
        score = torch.matmul(q_cal, k_cal) # [batch_size, head_num, max_q_len, max_kv_len]
        
        score = score / math.sqrt(size_per_head)
        mask = get_mask(input_lengths, context_lengths, max_q_len, max_kv_len, inoutput.dtype)
        score += mask
        
        score = torch.softmax(score, dim=-1)
        atten_out = torch.matmul(score, v_cal) # [batch_size, head_num, max_q_len, size_per_head]
        # update inoutput
        pre_batches_len = 0
        # inoutput: [token_num, hidden_units]
        atten_out = atten_out.permute(0, 2, 1, 3) # [batch_size, head_num, max_q_len, size_per_head] -> [batch_size, max_q_len, head_num, size_per_head]
        for batch_idx in range(batch_size):
            input_length = input_lengths[batch_idx].item()
            inoutput[pre_batches_len: pre_batches_len + input_length, :] = atten_out[batch_idx, :input_length, :, :].reshape(input_length, -1) # [len, head, dim] -> [len, head*dim]
            pre_batches_len += input_length
        return *key_cache, *value_cache, inoutput
    
    
    def fused_decoder_attention_inp(inoutput, qkv_weight, qkv_bias, key_cache, value_cache, batch_size, finished, total_padding_tokens, sequence_lengths, step, layer_id, local_head_num, local_kv_head_num, size_per_head, max_seq_len, rotary_embedding, rope_theta):
        qkv = torch.matmul(inoutput, qkv_weight) # [batch_size, hidden_units] * [hidden_units, (local_head_num+local_kv_head_num*2)*size_per_head]
        if qkv_bias is not None:
            qkv += qkv_bias
        q = qkv[:, :local_head_num*size_per_head].reshape(batch_size, local_head_num, size_per_head)  # [batch, head, dim]
        k = qkv[:, local_head_num*size_per_head: (local_head_num+local_kv_head_num)*size_per_head].reshape(batch_size, local_kv_head_num, size_per_head)
        v = qkv[:, (local_head_num+local_kv_head_num)*size_per_head:].reshape(batch_size, local_kv_head_num, size_per_head)
        
        def rope(x: torch.Tensor, timestamp: torch.Tensor, rotary_embedding: float, rope_theta: float) -> torch.Tensor:
            # x (torch.Tensor): [batch_size, head_num, size_per_head]
            batch_size = x.shape[0]
            head_num = x.shape[1]
            # input_len = x.shape[0]
            # head_num = x.shape[1]
            x = x.reshape(batch_size, head_num, size_per_head // 2, 2)
            # x_odd, x_even: [batch_size, head_num, size_per_head // 2, 1]
            x_even = x[:, :, :, 0].reshape(batch_size, head_num, size_per_head // 2, 1)
            x_odd = x[:, :, : ,1].reshape(batch_size, head_num, size_per_head // 2, 1)
            # theta: [size_per_head // 2], theta[i] = pow(rope_theta, -2 * i / rotary_embedding)
            theta = torch.pow(rope_theta, -2 * torch.arange(size_per_head // 2, dtype=torch.float32, device=x.device) / rotary_embedding)
            # timestamp: [batch_size]
            # sin, cos: [batch_size, 1, size_per_head // 2, 1]
            cos = torch.cos(timestamp.view(-1, 1) * theta.view(1, -1)).reshape(batch_size, 1, size_per_head // 2, 1)
            sin = torch.sin(timestamp.view(-1, 1) * theta.view(1, -1)).reshape(batch_size, 1, size_per_head // 2, 1)
            x_even_new = x_even * cos - x_odd * sin
            x_odd_new = x_odd * cos + x_even * sin
            x_new = torch.cat([x_even_new, x_odd_new], dim=-1).reshape(x.shape[0], x.shape[1], size_per_head)
            x_new = x_new.to(x.dtype)
            return x_new
        
        lengths = step - 1 - total_padding_tokens
        q = rope(q, lengths, rotary_embedding, rope_theta) 
        k = rope(k, lengths, rotary_embedding, rope_theta)
        for batch_idx in range(batch_size):
            if finished[batch_idx]:
                continue
            sequence_length = sequence_lengths[batch_idx].item()
            tlength = sequence_length
            first_step = max(0, tlength + 1 - max_seq_len)
            tlength_circ = tlength % max_seq_len
            
            # 1. q
            q_cal_i = q[batch_idx, :, :].reshape(1, local_head_num, size_per_head).permute(1, 0, 2)  #[head_num, 1, size_per_head] 
            
            # 2. k
            k_cache_i = key_cache[batch_idx][layer_id, :, :, :] # [head_num, max_seq_len, size_per_head]
            # 2.1 store to key_cache
            k_cache_i[:, tlength_circ, :] = k[batch_idx, :, :]
            # 2.2 get k_cal_i
            
            kvi_beg = first_step % max_seq_len
            # kvi_beg = first_step % max_seq_len = max(0, tlength + 1 - max_seq_len) % max_seq_len
            kvi_end = tlength % max_seq_len
            # kvi_end = tlength % max_seq_len;
            
            cat_kcal = []
            if kvi_beg > kvi_end:
                catki_cache_beg_gt = k_cache_i[:, kvi_beg:, :] # [head_num, max_seq_len - kvi_beg, size_per_head]
                cat_kcal.append(catki_cache_beg_gt)
                if kvi_end > 0:
                    catki_cache_end_gt = k_cache_i[:, :kvi_end, :] # [head_num, kvi_end, size_per_head]
                    cat_kcal.append(catki_cache_end_gt)
            elif kvi_beg < kvi_end:
                catki_cache_end_lt = k_cache_i[:, kvi_beg:kvi_end, :] # [head_num, kvi_end - kvi_beg, size_per_head]
                cat_kcal.append(catki_cache_end_lt)

            cat_kcal.append(k[batch_idx, :, :].reshape(local_kv_head_num, 1, size_per_head)) 
            
            k_cal_i = torch.cat(cat_kcal, dim=1) # [head_num, tlength - first_step + 1, size_per_head]
            k_cal_i = k_cal_i.permute(0, 2, 1)
            
            # 3. v
            v_cache_i = value_cache[batch_idx][layer_id, :, :, :] # [head_num, max_seq_len, size_per_head]
            # 3.1 store to value_cache
            v_cache_i[:, tlength_circ, :] = v[batch_idx, :, :]
            # 3.2 get v_cal_i
            cat_vcal = []
            if kvi_beg > kvi_end:
                catvi_cache_beg_gt = v_cache_i[:, kvi_beg:, :]
                cat_vcal.append(catvi_cache_beg_gt)
                if kvi_end > 0:
                    catvi_cache_end_gt = v_cache_i[:, :kvi_end, :]
                    cat_vcal.append(catvi_cache_end_gt)
            elif kvi_beg < kvi_end:
                catvi_cache_end_lt = v_cache_i[:, kvi_beg:kvi_end, :]
                cat_vcal.append(catvi_cache_end_lt)
                
            cat_vcal.append(v[batch_idx, :, :].reshape(local_kv_head_num, 1, size_per_head))
            
            v_cal_i = torch.cat(cat_vcal, dim=1) # [head_num, tlength - first_step + 1, size_per_head]
            
            # 4. cal
            qki =torch.matmul(q_cal_i, k_cal_i) # [head_num, 1, seq_len + 1]            
            qki = qki / math.sqrt(size_per_head)
            qki = torch.softmax(qki, dim=-1)
            inouti = torch.matmul(qki, v_cal_i) # [head_num, 1, size_per_head]
            inoutput[batch_idx, :] = inouti.reshape(local_head_num * size_per_head)
        
        return *key_cache, *value_cache, inoutput

    def fused_silu_ffn_inp(inoutput, weight1, weight2, weight3):
        matW1 = torch.matmul(inoutput, weight1) # [token_num, inter_size]
        matW1 = F.silu(matW1) # [token_num, inter_size]
        matW3 = torch.matmul(inoutput, weight3) # [token_num, inter_size]
        matW1 = matW1 * matW3 # [token_num, inter_size]
        inoutput = torch.matmul(matW1, weight2) # [token_num, hidden_units]
        return inoutput
    
    def setup_topk_runtime_args(top_ks, top_ps, skip_decode, batch_size, top_k, top_ks_size, top_p, top_ps_size):
        top_k_max = 1024
        for batch_idx in range(batch_size):
            k = top_ks[batch_idx].item() if top_ks_size > 1 else top_k
            p = top_ps[batch_idx].item() if top_ps_size > 1 else top_p
            
            if k == 0 and p == 0.0:
                k = 1
            if k > 0 and p == 0.0:
                p = 1.0

            top_ks[batch_idx] = min(k, top_k_max)
            top_ps[batch_idx] = max(min(p, 1.0), 0.0)
            skip_decode[batch_idx] = k == 0
            
        return top_ks, top_ps, skip_decode
    
    def setup_topp_runtime_args(top_ks, top_ps, skip_decode, batch_size, top_k, top_ks_size, top_p, top_ps_size, initial_top_p_buf, top_p_decay_buf, top_p_decay, top_p_min_buf, top_p_min, top_p_reset_ids_buf, top_p_reset_ids):
        top_p_decay_buf = torch.ones(batch_size, dtype=torch.float32) if top_p_decay is None else top_p_decay
        top_p_min_buf = torch.ones(batch_size, dtype=torch.float32) * 1e-6 if top_p_min is None else top_p_min
        top_p_reset_ids_buf = torch.ones(batch_size, dtype=torch.int64) * -1 if top_p_reset_ids is None else top_p_reset_ids
        
        for batch_idx in range(batch_size):
            k = top_ks[batch_idx].item() if top_ks_size > 1 else top_k
            p = top_ps[batch_idx].item() if top_ps_size > 1 else top_p
            if k == 0 and p == 0.0:
                k = 1
            top_ks[batch_idx] = k
            top_ps[batch_idx] = max(min(p, 1.0), 0.0)
            skip_decode[batch_idx] = k > 0
            
            if top_p_decay is not None and (top_p_decay_buf[batch_idx] > 1.0 or top_p_decay_buf[batch_idx] <= 0.0):
                top_p_decay_buf[batch_idx] = 1.0
            
            if top_p_min is not None and (top_p_min_buf[batch_idx] > 1.0 or top_p_min_buf[batch_idx] <= 0.0):
                top_p_min_buf[batch_idx] = 0.5
            
        initial_top_p_buf = top_ps.clone()
        
        return top_ks, top_ps, skip_decode, initial_top_p_buf, top_p_decay_buf, top_p_min_buf, top_p_reset_ids_buf
    
class GenOutputData(object):
    r'''
    Generate output data for all functions by using numpy and input data
    '''
    db_case_items = {}

    @staticmethod
    def run(diopi_item_config_path='diopi_case_items.cfg', input_path='data/inputs/',
            output_path='data/outputs/', fname='all_ops', model_name='diopi'):
        if not os.path.exists(input_path):
            logger.error("Input data is not generated!")
            sys.exit(0)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(diopi_item_config_path, 'rb') as f:
            all_cfg_dict = pickle.load(f)

        # XXX save case number in glob_var
        case_counter = 0
        func_name_list = []  # make the info log once

        for case_name in all_cfg_dict:
            each_cfg_dict = all_cfg_dict[case_name]
            func_name = each_cfg_dict["name"]
            item = {'case_name': case_name, 'model_name': model_name}
            if fname not in [func_name, 'all_ops']:
                continue
            data_path = os.path.join(input_path, case_name)
            input_ = get_data_from_file(data_path, case_name, 'input')
            if "no_output_ref" in each_cfg_dict:
                logger.info(f'diopi_functions.{func_name} [{case_name}] is set to no_output_ref, skip generate output')
                continue

            gen_tensor_obj = GenTensor(case_name, each_cfg_dict)

            try:
                output, saved_grads = gen_tensor_obj.gen_data(input_)
                item['result'] = 'passed'
            except Exception as err_msg:
                raise GenDataFailedException(f'Generate output data for diopi_functions.{func_name} [{case_name}] failed, cause by \n{err_msg}')
            GenOutputData.db_case_items[case_name] = item
            if output is not None:
                with open(os.path.join(output_path, case_name), "wb") as f:
                    pickle.dump(GenOutputData.to_numpy(output), f, protocol=4)
                    logger_str = "output"
                    case_counter += 1
                if saved_grads is not None:
                    saved_backward_pth = case_name.split(".pth")[0] + "_backward.pth"
                    with open(os.path.join(output_path, saved_backward_pth), "wb") as f:
                        pickle.dump(GenOutputData.to_numpy(saved_grads), f, protocol=4)
                    logger_str = f"{logger_str} and backward"

                if func_name not in func_name_list:
                    func_signature = f"diopi_functions.{func_name}"
                    logger.info(f"Generate benchmark {logger_str} data for {func_signature}")
                    func_name_list.append(func_name)

        logger.info(f"Generate test cases number for output data: {case_counter}")
        if case_counter == 0:
            logger.info("No benchmark output data is generated")
        else:
            logger.info("Generate benchmark output and backward data done!")

    @staticmethod
    def to_numpy(tensors):
        if isinstance(tensors, torch.Tensor):
            ndarrays = tensors.detach().cpu().numpy()
        elif isinstance(tensors, (list, tuple)):
            ndarrays = []
            for i in range(len(tensors)):
                if isinstance(tensors[i], torch.Tensor):
                    ndarrays.append(tensors[i].detach().cpu().numpy())
                else:
                    ndarrays.append(tensors[i])
        elif isinstance(tensors, dict):
            ndarrays = {}
            for k, v in tensors.items():
                if isinstance(v, torch.Tensor):
                    tmp = {k: v.detach().cpu().numpy()}
                else:
                    tmp = {k: v}
                ndarrays.update(tmp)
        elif isinstance(tensors, (int, float)):
            ndarrays = np.array(tensors)
        else:
            ndarrays = None

        return ndarrays


class GenTensor(object):
    def __init__(self, case_name, case_cfg) -> None:
        self.case_name = case_name
        self.case_cfg = case_cfg
        self.func_name = case_cfg["name"]
        self.module = "torch.nn.functional"
        self.input = None
        self.output = None
        self.if_forward_success = False

    def gen_data(self, input_data):
        output = self.gen_forward_data(input_data)
        saved_grads = self.gen_backward_data(input_data)
        return output, saved_grads

    def gen_forward_data(self, input_data):
        if self.case_cfg['interface']:
            self.module = self.case_cfg["interface"][0]
        function_paras = input_data["function_paras"]
        self.transfer_tensor_to_device(function_paras)
        kwargs = function_paras['kwargs']
        if self.module == "torch.Tensor":
            input = kwargs['input']
            self.input = input
            self.module = "input"
            del kwargs['input']
        if 'dtype' in kwargs.keys():
            kwargs['dtype'] = self.change_np_dtype_to_torch(kwargs['dtype'])
        func_call = f"{self.module}.{self.func_name}(**kwargs)"

        try:
            self.output = eval(func_call)
            self.if_forward_success = True
        except Exception as e:
            raise GenDataFailedException(f"Failed to execute function {func_call}, caused by {e}")
        return self.output

    def gen_backward_data(self, input_data):
        if not self.if_forward_success:
            return None
        function_paras = input_data["function_paras"]
        kwargs = function_paras['kwargs']
        saved_grads = None
        if function_paras["requires_grad"]:
            if self.module == "input":
                kwargs['input'] = self.input
            outputs = self.output
            if not isinstance(self.output, (list, tuple)):
                outputs = [self.output]

            requires_backward = self.case_cfg["requires_backward"]
            outputs_for_backward = outputs if len(requires_backward) == 0 \
                else [outputs[i] for i in requires_backward]

            inputs_name_for_grad, inputs_for_grad = self.get_name_and_data_for_grad(function_paras)
            if len(inputs_for_grad) != 0:
                grad_outputs = [torch.ones_like(i) for i in outputs_for_backward]
                grads = torch.autograd.grad(
                    outputs_for_backward, inputs_for_grad, grad_outputs, allow_unused=True)
                saved_grads = {k: v for k, v in zip(inputs_name_for_grad, grads)}
        return saved_grads

    def transfer_tensor_to_device(self, function_paras: dict):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for para in function_paras["kwargs"].keys():
            if isinstance(function_paras['kwargs'][para], np.ndarray):
                tensor = torch.from_numpy(function_paras['kwargs'][para])
                if function_paras["requires_grad"].get(para, []) == [True]:
                    tensor.requires_grad = True
                function_paras['kwargs'][para] = tensor.to(device=device)

            gen_policy = [i.get('gen_policy', None) for i in self.case_cfg['tensor_para']['args'] if i['ins'] == para]
            if_gen_list = len(gen_policy) > 0 and gen_policy[0] in GenPolicy.gen_list_policy
            if if_gen_list:
                if isinstance(function_paras['kwargs'][para], (list, tuple)):
                    tensors = function_paras['kwargs'][para]
                    for idx, ele in enumerate(tensors):
                        tensors[idx] = torch.from_numpy(ele).to(device=device)
                        if function_paras["requires_grad"].get(para, []) == [True]:
                            tensors[idx].requires_grad = True
                    function_paras['kwargs'][para] = tensors

    def get_name_and_data_for_grad(self, function_paras):
        inputs_for_grad_value = []
        inputs_for_grad_key = []
        for k, v in function_paras["kwargs"].items():
            if function_paras["requires_grad"].get(k, []) == [True]:
                inputs_for_grad_key.append(k)
                if isinstance(v, (list, tuple)):
                    inputs_for_grad_value.extend(v)
                else:
                    inputs_for_grad_value.append(v)
        return inputs_for_grad_key, inputs_for_grad_value

    def change_np_dtype_to_torch(self, dtype):
        if dtype == np.bool_:
            return torch.bool
        return eval(str(dtype).replace("<class 'numpy.", "torch.").replace("'>", ""))


if __name__ == '__main__':
    GenOutputData.run(os.path.join(os.path.dirname(__file__), '../cache/diopi_case_items.cfg'),
                      os.path.join(os.path.dirname(__file__), '../cache/data/inputs/'),
                      os.path.join(os.path.dirname(__file__), '../cache/data/outputs/'))
