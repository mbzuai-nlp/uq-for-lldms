import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import os
import sys
import random
import copy
import math
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from typing import Optional, Dict, Any, Tuple, List

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


import torch
import torch.nn.functional as F


@torch.no_grad()
def generate_refusion(model, tokenizer, prompt, gen_length=128, temperature=0., mask_id=151670, slot_size=8,
             model_path='', serial_num_blocks=2, slot_threshold=0.9, token_threshold=0.9):

    if prompt.shape[0] != 1:
        raise ValueError("ReFusion generation currently supports batch_size=1.")

    slot_threshold = slot_threshold
    token_threshold = token_threshold
    original_gen_length = gen_length
    sum_TPF = 0.0
    forward_count = 0
    nfe = 0

    eos_token_ids = []
    configured_eos_values = (
        tokenizer.eos_token_id,
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    )
    for configured_eos in configured_eos_values:
        if configured_eos is None:
            continue
        if not isinstance(configured_eos, (list, tuple, set)):
            configured_eos = [configured_eos]
        eos_token_ids.extend(int(token_id) for token_id in configured_eos)
    eos_token_ids = tuple(dict.fromkeys(eos_token_ids))
    if not eos_token_ids:
        raise ValueError("ReFusion generation requires at least one EOS token ID.")

    def matches_eos(tokens):
        result = torch.zeros_like(tokens, dtype=torch.bool)
        for token_id in eos_token_ids:
            result |= tokens == token_id
        return result

    batch_size = 1
    prompt_len = prompt.shape[1]
    device = model.device

    gen_pad_len = (slot_size - (gen_length % slot_size)) % slot_size
    serial_num_blocks = _serial_blocks_for_length(gen_length, slot_size, serial_num_blocks)
    gen_length = gen_length + gen_pad_len
    gen_x = torch.full((batch_size, gen_length), mask_id, dtype=torch.long, device=device)

    prompt_pos_ids = torch.arange(prompt_len, dtype=torch.long, device=device).unsqueeze(0)
    gen_pos_ids = torch.arange(prompt_len, prompt_len + gen_length, dtype=torch.long, device=device).unsqueeze(0)

    cur_x = prompt.clone()
    cur_pos = prompt_pos_ids.clone()

    cur_slot_size = slot_size

    eos_flag = False
    block_length = gen_length // serial_num_blocks

    past_key_values = None

    committed_mask = torch.zeros((batch_size, gen_length), dtype=torch.bool, device=device)
    first_transfer_step = torch.full((batch_size, prompt_len + gen_length), -1, dtype=torch.long, device=device)
    confidence_at_commit = torch.full((batch_size, prompt_len + gen_length), float("-inf"), device=device)
    mask_fraction_at_commit = torch.full((batch_size, prompt_len + gen_length), float("-inf"), device=device)
    gen_trace = [[] for _ in range(serial_num_blocks)]
    mask_trace = [[] for _ in range(serial_num_blocks)]
    gen_trace_logits = [[] for _ in range(serial_num_blocks)]
    gen_trace_entropy = [[] for _ in range(serial_num_blocks)]
    gen_trace_argmax = [[] for _ in range(serial_num_blocks)]
    gen_trace_top2gap = [[] for _ in range(serial_num_blocks)]
    global_step = 0

    def track_step(block_idx, pos_ids, logits):
        s = block_idx * block_length
        e = s + block_length
        mask_trace[block_idx].append((~committed_mask[:, s:e]).clone())

        pos_ids = pos_ids.reshape(-1)
        logits = logits.reshape(-1, logits.shape[-1])
        rel_pos = pos_ids - prompt_len
        valid = (rel_pos >= s) & (rel_pos < e)

        trace_dtype = logits.dtype
        step_logits = torch.full((batch_size, block_length), float("-inf"), dtype=trace_dtype, device=device)
        step_entropy = torch.full((batch_size, block_length), float("-inf"), dtype=trace_dtype, device=device)
        step_argmax = torch.full((batch_size, block_length), -1, dtype=torch.long, device=device)
        step_top2gap = torch.full((batch_size, block_length), float("-inf"), dtype=trace_dtype, device=device)

        if torch.any(valid):
            block_pos = rel_pos[valid] - s
            step_logits_raw = logits[valid]
            log_probs = step_logits_raw.log_softmax(-1)
            probs = step_logits_raw.softmax(-1)
            step_logits[0, block_pos] = log_probs.max(-1).values
            step_entropy[0, block_pos] = -(probs * probs.log()).sum(-1)
            step_argmax[0, block_pos] = step_logits_raw.argmax(-1)
            top2 = probs.topk(2, dim=-1).values
            step_top2gap[0, block_pos] = top2[:, 0] - top2[:, 1]

        gen_trace_logits[block_idx].append(step_logits)
        gen_trace_entropy[block_idx].append(step_entropy)
        gen_trace_argmax[block_idx].append(step_argmax)
        gen_trace_top2gap[block_idx].append(step_top2gap)

    def track_commit(tokens, pos_ids, logits):
        nonlocal confidence_at_commit, mask_fraction_at_commit
        if tokens.numel() == 0:
            return
        tokens = tokens.reshape(-1)
        pos_ids = pos_ids.reshape(-1)
        logits = logits.reshape(-1, logits.shape[-1])
        rel_pos = pos_ids - prompt_len
        valid = (rel_pos >= 0) & (rel_pos < gen_length)
        if not torch.any(valid):
            return

        tokens = tokens[valid]
        rel_pos = rel_pos[valid]
        logits = logits[valid]
        log_probs = logits.log_softmax(-1)
        max_log_probs = log_probs.max(-1).values

        full_pos = rel_pos + prompt_len
        is_new = ~committed_mask[0, rel_pos]
        if torch.any(is_new):
            new_rel = rel_pos[is_new]
            new_full = full_pos[is_new]
            committed_mask[0, new_rel] = True
            first_transfer_step[0, new_full] = global_step
            newly_committed = torch.zeros_like(first_transfer_step, dtype=torch.bool)
            newly_committed[0, new_full] = True
            committed_log_probs = torch.full(
                confidence_at_commit.shape,
                float("-inf"),
                dtype=max_log_probs.dtype,
                device=device,
            )
            committed_log_probs[0, new_full] = max_log_probs[is_new]
            confidence_at_commit = torch.where(
                newly_committed, committed_log_probs, confidence_at_commit,
            )
            remaining = (~committed_mask).float().sum(-1, keepdim=True) / (prompt_len + gen_length)
            mask_fraction_at_commit = torch.where(
                newly_committed,
                remaining.expand_as(newly_committed),
                mask_fraction_at_commit,
            )

    def track_text(block_idx):
        # Slots can be committed out of order. Store the local block trace in
        # generation-position order, matching the trace contract used by LLaDA.
        block_start = block_idx * block_length
        block_end = min(block_start + block_length, original_gen_length)
        rel_pos = cur_pos - prompt_len
        block_texts = []
        for batch_idx in range(batch_size):
            in_block = (
                (rel_pos[batch_idx] >= block_start)
                & (rel_pos[batch_idx] < block_end)
            )
            positions = rel_pos[batch_idx][in_block]
            tokens = cur_x[batch_idx][in_block]
            if tokens.numel() > 0:
                tokens = tokens[torch.argsort(positions)]
            block_texts.append(
                tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
            )
        gen_trace[block_idx].append(block_texts)

    def cache_seq_len(cache):
        if hasattr(cache, "get_seq_length"):
            return cache.get_seq_length()
        return cache[0][0].shape[-2]


    for serial_num_block in range(serial_num_blocks):

        # block level
        cur_gen_x = gen_x[:, serial_num_block*block_length:(serial_num_block+1)*block_length] # (batch_size, block_length)
        cur_gen_pos_ids = gen_pos_ids[:, serial_num_block*block_length:(serial_num_block+1)*block_length] # (batch_size, block_length)

        cur_gen_blocks_x = cur_gen_x.reshape(batch_size, -1, cur_slot_size)
        cur_gen_blocks_pos_ids = cur_gen_pos_ids.reshape(batch_size, -1, cur_slot_size)

        # slot level generation
        while cur_gen_blocks_x.numel() > 0:
            cur_gen_blocks_x = cur_gen_blocks_x.reshape(batch_size, -1, cur_slot_size)
            cur_gen_blocks_pos_ids = cur_gen_blocks_pos_ids.reshape(batch_size, -1, cur_slot_size)

            flat_gen_blocks_x = cur_gen_blocks_x.view(batch_size, -1)
            flat_gen_blocks_pos_ids = cur_gen_blocks_pos_ids.view(batch_size, -1)

            prefix_block_tag = False
            
            # MDM
            if past_key_values is None:
                input_x = torch.cat((cur_x, flat_gen_blocks_x), dim=1)
                input_pos_ids = torch.cat((cur_pos, flat_gen_blocks_pos_ids), dim=1)
                outputs = model(
                    input_ids=input_x,
                    position_ids=input_pos_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                nfe += 1
            else:
                outputs = model(
                    input_ids=flat_gen_blocks_x,
                    position_ids=flat_gen_blocks_pos_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                nfe += 1

            logits = outputs.logits

            gen_logits = logits[:, -flat_gen_blocks_x.shape[1]:, :]
            track_step(serial_num_block, flat_gen_blocks_pos_ids, gen_logits)

            past_key_values = outputs.past_key_values
            past_key_values.crop(cur_x.shape[1])
            assert cur_x.shape[-1] == cache_seq_len(past_key_values)

            logits_with_noise = add_gumbel_noise(gen_logits, temperature=temperature)
            x0_gen = torch.argmax(logits_with_noise, dim=-1)
            x0_gen_blocks = x0_gen.view(batch_size, -1, cur_slot_size)

            p_softmax = F.softmax(gen_logits, dim=-1)
            x0_p_softmax = torch.gather(p_softmax, dim=-1, index=torch.unsqueeze(x0_gen, -1)).squeeze(-1)
            
            x0_p_softmax_blocks = x0_p_softmax.view(batch_size, -1, cur_slot_size)
            block_confidence_softmax = x0_p_softmax_blocks[:,:,0] # (bsz, num_slots)
            
            is_confident_block = block_confidence_softmax > slot_threshold
            counts_block = torch.sum(is_confident_block, dim=1).item()
            topk_indices_relative = is_confident_block[0].nonzero(as_tuple=True)[0]

            if counts_block <= 0:
                counts_block = 1
                _, topk_indices_relative = torch.topk(block_confidence_softmax.squeeze(0), k=1)

            # choose slot
            topk_indices_relative, _ = torch.sort(topk_indices_relative)

            chosen_gen_blocks = x0_gen_blocks[0, topk_indices_relative, :]
            chosen_position_ids = cur_gen_blocks_pos_ids[0, topk_indices_relative, :]
            chosen_p_softmax_blocks = x0_p_softmax_blocks[0, topk_indices_relative, :]


            # Global Verification            
            outputs = model(
                input_ids=chosen_gen_blocks.reshape(1, -1),
                position_ids=chosen_position_ids.reshape(1, -1),
                past_key_values=past_key_values,
                use_cache=True,
            )
            nfe += 1

            AR_logits = outputs.logits #[1, len, vocab_len]
            AR_logits = torch.cat([AR_logits[:,:1], AR_logits[:, :-1]], dim=1)
            AR_p_softmax = F.softmax(AR_logits, dim=-1) #[1, len, 1]
            AR_x0_p_softmax = torch.gather(AR_p_softmax, dim=-1, index=torch.unsqueeze(chosen_gen_blocks.reshape(1, -1), -1)).squeeze(-1) #[1, len]
            AR_x0_p_softmax_blocks = AR_x0_p_softmax.reshape(-1, cur_slot_size)
            chosen_p_softmax_blocks[:,1:] = AR_x0_p_softmax_blocks[:,1:]

            
            prob_mask = chosen_p_softmax_blocks > token_threshold
            prob_mask[:, 0] = 1
            tag_blocks = torch.cumprod(prob_mask.int(), dim=-1)

            tag_tokens = torch.cumprod(prob_mask.int().reshape(1, -1), dim=-1)
            prefix_len = torch.sum(tag_tokens, dim=-1).item()
            flat_chosen_gen_blocks = chosen_gen_blocks.reshape(1, -1)
            confident_prefix_tokens = flat_chosen_gen_blocks[:, :prefix_len]

            if prefix_len > 0:
                is_eos_in_prefix = matches_eos(confident_prefix_tokens.squeeze(0))
                eos_found_flag = torch.any(is_eos_in_prefix).item()

                remain_indices = []

                indices_to_remove = set()

                if eos_found_flag:
                    first_eos_pos_tensor = torch.argmax(is_eos_in_prefix.int()).item()

                    eos_block_pos = first_eos_pos_tensor // cur_slot_size + 1
                    eos_token_pos = first_eos_pos_tensor - (first_eos_pos_tensor // cur_slot_size) * cur_slot_size

                    eos_block = topk_indices_relative[eos_block_pos-1].item()

                    remain_indices.extend(topk_indices_relative[:eos_block_pos].tolist())
                    
                    topk_indices_relative = torch.tensor([], dtype=torch.long, device=device)

                    eos_flag = True

                    indices_after_eos = list(range(eos_block, cur_gen_blocks_x.shape[1]))
                    indices_to_remove.update(indices_after_eos)

                elif (prefix_len // cur_slot_size) > 0:
                    num_prefix_blocks = prefix_len // cur_slot_size
                    remain_indices.extend(topk_indices_relative[:num_prefix_blocks].tolist())

                    topk_indices_relative = topk_indices_relative[num_prefix_blocks:]
                    tag_blocks = tag_blocks[num_prefix_blocks:]
            
                if len(remain_indices) > 0:

                    indices_to_remove.update(remain_indices)

                    token_indices = []

                    for i_idx, b_idx in enumerate(remain_indices):
                        start_index = b_idx * cur_slot_size
                        
                        current_block_len = cur_slot_size
                        # If EOS exists and this is the last slot, then adjust the length.
                        if eos_found_flag and i_idx == len(remain_indices) - 1:
                            current_block_len = eos_token_pos + 1
                            
                        
                        end_index = start_index + current_block_len
                        block_range = torch.arange(start_index, end_index, dtype=torch.long, device=device)
                        
                        token_indices.append(block_range)

                    full_token_indices = torch.cat(token_indices)

                    commit_tokens = x0_gen[:, full_token_indices]
                    commit_pos_ids = flat_gen_blocks_pos_ids[:, full_token_indices]
                    commit_logits = gen_logits[:, full_token_indices, :]
                    cur_x = torch.cat((cur_x, commit_tokens), dim=1)
                    cur_pos = torch.cat((cur_pos, commit_pos_ids), dim=1)
                    track_commit(commit_tokens, commit_pos_ids, commit_logits)
                    track_text(serial_num_block)
                    global_step += 1

                    past_key_values = outputs.past_key_values
                    past_key_values.crop(cur_x.shape[1])
                    
                    assert cur_x.shape[-1] == cache_seq_len(past_key_values)

                    prefix_block_tag = True

                    sum_TPF += cur_slot_size * len(remain_indices) / 2
                    forward_count += 1

            if prefix_block_tag == True:
                keep_mask = torch.ones(cur_gen_blocks_x.shape[1], dtype=torch.bool, device=device)
                keep_mask[list(indices_to_remove)] = False
                cur_gen_blocks_x = cur_gen_blocks_x[:, keep_mask, :]
                cur_gen_blocks_pos_ids = cur_gen_blocks_pos_ids[:, keep_mask, :]

                continue

            elif prefix_block_tag == False:
                past_key_values = outputs.past_key_values
                past_key_values.crop(cur_x.shape[1])
                assert cur_x.shape[-1] == cache_seq_len(past_key_values)

            indices_to_remove = set(topk_indices_relative.tolist())

            current_speculative_blocks = chosen_gen_blocks.clone()
            accepted_prefix_len = 0
            eos_found_in_loop = False

            initial_verified_mask = tag_blocks.bool().reshape(-1)
            if torch.any(initial_verified_mask):
                flat_chosen_gen_blocks = chosen_gen_blocks.reshape(-1)
                flat_chosen_position_ids = chosen_position_ids.reshape(-1)
                flat_AR_logits = AR_logits.reshape(-1, AR_logits.shape[-1])
                track_commit(
                    flat_chosen_gen_blocks[initial_verified_mask].reshape(1, -1),
                    flat_chosen_position_ids[initial_verified_mask].reshape(1, -1),
                    flat_AR_logits[initial_verified_mask].reshape(1, -1, AR_logits.shape[-1]),
                )
            global_step += 1

            if past_key_values is not None and counts_block > 1:
                past_key_values.batch_repeat_interleave(counts_block)
            
            for loop_iter in range(cur_slot_size):
                if not torch.any(tag_blocks == 0):
                    break

                input_tokens = current_speculative_blocks[:, accepted_prefix_len:]
                input_pos = chosen_position_ids[:, accepted_prefix_len:]

                current_tags = tag_blocks[:, accepted_prefix_len:]
                masked_input_tokens = torch.where(current_tags.bool(), input_tokens, mask_id)

                # Prediction
                draft_len = cache_seq_len(past_key_values)
                draft_outputs = model(
                    input_ids=masked_input_tokens,
                    position_ids=input_pos,
                    past_key_values=past_key_values,
                    use_cache=False,
                )
                nfe += 1
                past_key_values.crop(draft_len)
                draft_logits = draft_outputs.logits
                proposed_tokens = torch.argmax(draft_logits, dim=-1)

                input_tokens = torch.where(current_tags.bool(), input_tokens, proposed_tokens)
                current_speculative_blocks[:, accepted_prefix_len:] = input_tokens

                # Verification
                verify_outputs = model(
                    input_ids=input_tokens,
                    position_ids=input_pos,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                nfe += 1
                verify_logits = verify_outputs.logits
                verify_logits = torch.cat([verify_logits[:,:1], verify_logits[:, :-1]], dim=1)
                track_step(serial_num_block, input_pos, verify_logits)
                
                verify_probs = F.softmax(verify_logits, dim=-1)
                gathered_probs = torch.gather(verify_probs, -1, input_tokens.unsqueeze(-1)).squeeze(-1)
                
                prob_mask = gathered_probs > token_threshold

                # Keep at least one token
                update_tag_blocks = F.pad(tag_blocks[:, accepted_prefix_len:], (1, 0), value=1)[:, :-1]

                prob_mask[update_tag_blocks == 1] = True

                old_tags = tag_blocks[:, accepted_prefix_len:].clone()
                new_tags = torch.cumprod(prob_mask.int(), dim=-1)
                tag_blocks[:, accepted_prefix_len:] = new_tags

                newly_verified_mask = (new_tags == 1) & (old_tags == 0)
                if torch.any(newly_verified_mask):
                    track_commit(
                        input_tokens[newly_verified_mask].reshape(1, -1),
                        input_pos[newly_verified_mask].reshape(1, -1),
                        verify_logits[newly_verified_mask].reshape(1, -1, verify_logits.shape[-1]),
                    )
                global_step += 1

                newly_verified_mask = (tag_blocks[:, accepted_prefix_len:] == 1)
                is_eos_in_new = matches_eos(current_speculative_blocks[:, accepted_prefix_len:]) & newly_verified_mask

                if torch.any(is_eos_in_new):
                    eos_found_in_loop = True
                    first_eos_block_idx = torch.where(torch.any(is_eos_in_new, dim=1))[0][0].item()
                    eos_block_idx = topk_indices_relative[first_eos_block_idx].item()
                    
                    current_speculative_blocks = current_speculative_blocks[:first_eos_block_idx+1]
                    tag_blocks = tag_blocks[:first_eos_block_idx+1]
                    tag_blocks[first_eos_block_idx] = 1
                    chosen_position_ids = chosen_position_ids[:first_eos_block_idx+1]
                    topk_indices_relative = topk_indices_relative[:first_eos_block_idx+1]
                    if verify_outputs.past_key_values is not None:
                         verify_outputs.past_key_values.batch_select_minibatch(first_eos_block_idx + 1)

                current_tags = tag_blocks[:, accepted_prefix_len:]
                len_per_block = torch.sum(current_tags, dim=1)
                newly_accepted_len = torch.min(len_per_block).item()
                if newly_accepted_len > 0:
                    if torch.any(tag_blocks == 0):
                        accepted_prefix_len = accepted_prefix_len + newly_accepted_len - 1 
                    else:
                        accepted_prefix_len = accepted_prefix_len + newly_accepted_len
                    past_key_values = verify_outputs.past_key_values
                    if past_key_values is not None:
                        past_key_values.crop(cur_x.shape[1] + accepted_prefix_len)
                    
            sum_TPF += (cur_slot_size * counts_block) / (loop_iter * 2 + 2)
            forward_count += 1

            ar_kv_cache = tuple(
                (
                    key[:, :, -cur_slot_size:, :],  # key
                    value[:, :, -cur_slot_size:, :]   # value
                )
                for key, value, *_ in past_key_values
            )


            past_key_values.crop(cur_x.shape[1])
            past_key_values.batch_select_indices(torch.tensor([0], dtype=torch.long))

            eos_mask = matches_eos(current_speculative_blocks) # (k*cur_slot_size)
            keep_mask = (torch.cumsum(eos_mask.flatten().int(), dim=-1) - eos_mask.flatten().int()) == 0
            kept_tokens = current_speculative_blocks.flatten()[keep_mask].reshape(batch_size, -1)
            kept_pos_ids = chosen_position_ids.flatten()[keep_mask].reshape(batch_size, -1)

            # update KV cache
            if kept_tokens.numel() > 0 and ar_kv_cache is not None:
                new_past = []
                for i, (key, val) in enumerate(ar_kv_cache):
                    num_heads, _, head_dim = key.shape[1], key.shape[2], key.shape[3]
                    
                    flat_key = key.permute(1, 0, 2, 3).reshape(1, num_heads, -1, head_dim)
                    flat_val = val.permute(1, 0, 2, 3).reshape(1, num_heads, -1, head_dim)

                    layer_keep_mask = keep_mask.to(flat_key.device)
                    kept_key = flat_key[:, :, layer_keep_mask, :]
                    kept_val = flat_val[:, :, layer_keep_mask, :]
                
                    new_past.append((kept_key, kept_val))

                kept_kv = tuple(new_past)

            past_key_values.full_update(kept_kv)

            cur_x = torch.cat((cur_x, kept_tokens), dim=1)
            cur_pos = torch.cat((cur_pos, kept_pos_ids), dim=1)
            track_text(serial_num_block)

            assert cur_x.shape[-1] == cache_seq_len(past_key_values)

            if eos_found_in_loop:
                indices_after_eos = list(range(eos_block_idx, cur_gen_blocks_x.shape[1]))
                indices_to_remove.update(indices_after_eos)
                eos_flag = True

            keep_mask = torch.ones(cur_gen_blocks_x.shape[1], dtype=torch.bool, device=device)
            keep_mask[list(indices_to_remove)] = False
            cur_gen_blocks_x = cur_gen_blocks_x[:, keep_mask, :]
            cur_gen_blocks_pos_ids = cur_gen_blocks_pos_ids[:, keep_mask, :]

        if eos_flag:
            break

    _, re_mask_indices = torch.sort(cur_pos, dim=-1)
    x = torch.gather(cur_x, dim=-1, index=re_mask_indices)
    x = x[:, : prompt_len + original_gen_length]

    gen_len = x.shape[1] - prompt_len
    trace_slice = slice(0, gen_len)

    def pack_trace(values, padding_value, dtype=None):
        sample = next((inner[0] for inner in values if len(inner)), None)
        if sample is None:
            dtype = dtype or torch.float32
            return torch.empty((batch_size, serial_num_blocks, 0, block_length), dtype=dtype, device=device)
        dtype = dtype or sample.dtype
        stacked = [
            torch.stack(inner, dim=0)
            if len(inner)
            else torch.empty((0, batch_size, block_length), dtype=dtype, device=device)
            for inner in values
        ]
        padded = torch.nn.utils.rnn.pad_sequence(
            stacked, batch_first=True, padding_value=padding_value
        )
        return padded.permute(2, 0, 1, 3)

    gen_trace_logits = pack_trace(gen_trace_logits, float("-inf"))
    gen_trace_entropy = pack_trace(gen_trace_entropy, float("-inf"))
    gen_trace_argmax = pack_trace(gen_trace_argmax, -1, dtype=torch.long)
    gen_trace_top2gap = pack_trace(gen_trace_top2gap, float("-inf"))

    mask_trace_counts = torch.stack(
        [
            torch.stack(inner, dim=0).int().sum(0)
            if len(inner)
            else torch.zeros((batch_size, block_length), dtype=torch.int, device=device)
            for inner in mask_trace
        ],
        dim=1,
    )
    remask_count = mask_trace_counts.reshape(batch_size, -1)[:, trace_slice]

    # Match LLaDA's trace views: local per-block traces, the last meaningful
    # block, that block with its finalized prefix, and the full cumulative trace.
    last_trace = next(
        (
            (block_idx, block_trace)
            for block_idx, block_trace in reversed(list(enumerate(gen_trace)))
            if len(block_trace) > 1
        ),
        None,
    )
    if last_trace is None:
        generation_trace = [
            tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)
        ]
        gen_trace_last = generation_trace
    else:
        last_block_idx, gen_trace_last = last_trace
        finalized_prefix = [
            "".join(
                gen_trace[block_idx][-1][batch_idx]
                for block_idx in range(last_block_idx)
                if gen_trace[block_idx]
            )
            for batch_idx in range(batch_size)
        ]
        generation_trace = [
            [
                finalized_prefix[batch_idx] + step_trace[batch_idx]
                for batch_idx in range(batch_size)
            ]
            for step_trace in gen_trace_last
        ]

    full_generation_trace = []
    finalized_prefix = [""] * batch_size
    for block_trace in gen_trace:
        if not block_trace:
            continue
        for step_trace in block_trace:
            full_generation_trace.append(
                [
                    finalized_prefix[batch_idx] + step_trace[batch_idx]
                    for batch_idx in range(batch_size)
                ]
            )
        finalized_prefix = [
            finalized_prefix[batch_idx] + block_trace[-1][batch_idx]
            for batch_idx in range(batch_size)
        ]

    remask_stats = {
        "generation_trace": generation_trace,
        "full_generation_trace": full_generation_trace,
        "last_block_generation_trace": gen_trace_last,
        "block_generation_traces": gen_trace,
        "remask_count": remask_count,
        "first_remask_step": first_transfer_step[:, : prompt_len + gen_len],
        "gen_trace_logits": gen_trace_logits,
        "gen_trace_entropy": gen_trace_entropy,
        "gen_trace_argmax": gen_trace_argmax,
        "gen_trace_top2gap": gen_trace_top2gap,
        "confidence_at_commit": confidence_at_commit[:, : prompt_len + gen_len],
        "mask_fraction_at_commit": mask_fraction_at_commit[:, : prompt_len + gen_len],
    }

    return x, nfe, remask_stats


def _serial_blocks_for_length(gen_length, slot_size, serial_num_blocks):
    gen_pad_len = (slot_size - (gen_length % slot_size)) % slot_size
    num_slots = max(1, (gen_length + gen_pad_len) // slot_size)
    serial_num_blocks = min(max(1, serial_num_blocks), num_slots)
    while num_slots % serial_num_blocks != 0:
        serial_num_blocks -= 1
    return serial_num_blocks
