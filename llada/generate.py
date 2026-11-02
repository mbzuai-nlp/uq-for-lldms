# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

from typing import Any
import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel

from torch.cuda import nvtx


def build_block_attention_mask(batch_size, seq_length, device, dtype=torch.bfloat16):
    """Build a full-attention 4D mask of shape (B, 1, L, L) with all ones.

    LLaDA 2.1 requires a 4D block attention mask. For diffusion models,
    we use bidirectional (full) attention so every token attends to every
    other token.
    """
    return torch.ones(
        (batch_size, 1, seq_length, seq_length), device=device, dtype=dtype
    )


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def add_gumbel_noise_optimized(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits

    logits = logits.to(torch.float64)

    # Reuse the noise buffer for gumbel computation instead of allocating intermediates
    noise = torch.rand_like(logits)
    noise.log_().neg_().pow_(temperature)  # in-place: rand → log → neg → pow

    # One allocation for the result, then in-place divide
    return logits.exp().div_(noise)


# def get_num_transfer_tokens(mask_index, steps):
#     '''
#     In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
#     Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
#     the expected number of tokens transitioned at each step should be consistent.

#     This function is designed to precompute the number of tokens that need to be transitioned at each step.
#     '''
#     mask_num = mask_index.sum(dim=1, keepdim=True)

#     base = mask_num // steps
#     remainder = mask_num % steps

#     num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

#     for i in range(mask_num.size(0)):
#         num_transfer_tokens[i, :remainder[i]] += 1

#     return num_transfer_tokens


def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    block_mask_index: (B, L) bool – which positions are masked in the current block
    returns: (B, steps) int – how many tokens to transfer at each step per batch item
    """
    device = block_mask_index.device
    dtype = torch.long

    total = block_mask_index.sum(dim=1)  # (B,)
    base = torch.div(total, steps, rounding_mode="floor")  # (B,)
    rem = total - base * steps  # (B,)

    # Start with base for all steps
    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(dtype)  # (B, steps)

    # Add +1 to the first `rem[b]` steps for each batch b — without tensor slicing
    cols = torch.arange(steps, device=device).unsqueeze(0)  # (1, steps)
    add_mask = cols < rem.unsqueeze(1)  # (B, steps)
    num_transfer_tokens = num_transfer_tokens + add_mask.to(dtype)  # (B, steps)

    return num_transfer_tokens


@torch.no_grad()
def generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
    factor=None,
    tokenizer=None,
    needs_attention_mask=False,
    shift_logits=False,
):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of[MASK] is 126336.
    """
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long
    ).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    # Statistics: track remasking count for each token position
    remask_count = torch.zeros(
        (B, Lp + gen_length), dtype=torch.long, device=model.device
    )
    # Track the step at which each token was first transferred (unmasked)
    first_transfer_step = torch.full(
        (B, Lp + gen_length), -1, dtype=torch.long, device=model.device
    )
    confidence_at_commit = torch.full(
        (B, Lp + gen_length), float('-inf'), device=model.device
    )
    mask_fraction_at_commit = torch.full(
        (B, Lp + gen_length), float('-inf'), device=model.device
    )
    gen_trace = []
    mask_trace = []
    gen_trace_logits = []
    gen_trace_entropy = []
    gen_trace_argmax = []
    gen_trace_top2gap = []

    nfe = 0
    global_step = 0
    for num_block in range(num_blocks):
        gen_trace.append([])
        mask_trace.append([])
        gen_trace_logits.append([])
        gen_trace_entropy.append([])
        gen_trace_argmax.append([])
        gen_trace_top2gap.append([])
        s = prompt.shape[1] + num_block * block_length
        e = s + block_length
        block_mask_index = (
            x[
                :,
                prompt.shape[1]
                + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = x == mask_id
            model_kwargs = {}
            if needs_attention_mask:
                model_kwargs["attention_mask"] = build_block_attention_mask(
                    x.shape[0], x.shape[1], x.device
                )
            logits = model(x, **model_kwargs).logits
            # Dream: logit at position i predicts token i+1. No-op when
            # shift_logits=False, so the LLaDA no-cache path is unchanged.
            if shift_logits:
                logits = apply_shift_logits(logits)
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length :] = 0
            if factor is None:
                x0, transfer_index = get_transfer_index(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x,
                    num_transfer_tokens[:, i] if threshold is None else None,
                    threshold,
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(
                    logits, temperature, remasking, mask_index, x, None, factor
                )

            # Track logits, entropy, and argmax for the generated block
            if tokenizer is not None:
                logits_block = logits[:, s:e]
                gen_trace_logits[-1].append(
                    logits_block.log_softmax(-1).max(-1).values
                )
                probs_block = logits_block.softmax(-1)
                entropy_block = -(probs_block * probs_block.log()).sum(-1)
                gen_trace_entropy[-1].append(entropy_block)
                gen_trace_argmax[-1].append(logits_block.argmax(-1))
                top2 = probs_block.topk(2, dim=-1).values  # (B, block_length, 2)
                gen_trace_top2gap[-1].append(top2[:, :, 0] - top2[:, :, 1])

            # Track generation and mask traces if tokenizer is provided
            if tokenizer is not None:
                gen_trace[-1].append(
                    tokenizer.batch_decode(x0[:, s:e], skip_special_tokens=True)
                )
                mask_trace[-1].append(x[:, s:e] == mask_id)

            x[transfer_index] = x0[transfer_index]

            # Update remasking statistics
            remask_count = remask_count + transfer_index.long()
            # Track first transfer step and confidence/mask-fraction at commitment
            # Exclude prompt positions — only track generated tokens
            gen_mask = torch.zeros_like(transfer_index)
            gen_mask[:, Lp:] = True
            newly_committed = transfer_index & (first_transfer_step == -1) & gen_mask
            first_transfer_step = torch.where(
                newly_committed, global_step, first_transfer_step,
            )
            if tokenizer is not None:
                # Confidence at commitment: log-prob of the committed token
                step_logprobs = logits.log_softmax(-1).max(-1).values
                confidence_at_commit = torch.where(
                    newly_committed, step_logprobs, confidence_at_commit,
                )
                # Mask fraction at commitment: what fraction of sequence is still masked
                current_mask_frac = (x == mask_id).float().sum(-1, keepdim=True) / x.shape[1]
                mask_fraction_at_commit = torch.where(
                    newly_committed,
                    current_mask_frac.expand_as(newly_committed),
                    mask_fraction_at_commit,
                )
            global_step += 1

            i += 1
            if (
                x[
                    :,
                    prompt.shape[1]
                    + num_block * block_length : prompt.shape[1]
                    + (num_block + 1) * block_length,
                ]
                == mask_id
            ).sum() == 0:
                break

    # Compute remask_stats if tokenizer was provided
    if tokenizer is not None:
        mask_trace = torch.stack(
            [torch.stack(inner, dim=0).int().sum(0) for inner in mask_trace], dim=0
        )
        mask_trace = mask_trace.permute(1, 0, 2)
        mask_trace = mask_trace.reshape(mask_trace.shape[0], -1)

        # Original behavior: trace only for the last non-empty block
        gen_trace_last = None  # Initialize before try/except
        try:
            from_end_index, gen_trace_last = next(
                (i, x) for i, x in enumerate(reversed(gen_trace)) if len(x) > 1
            )
            last_block = len(gen_trace) - from_end_index - 1
            last_step_trace = [
                "".join(
                    gen_trace[block_i][-1][batch_i] for block_i in range(last_block)
                )
                for batch_i in range(len(gen_trace[0][1]))
            ]
            if len(last_step_trace):
                generation_trace = [
                    [
                        (
                            last_step_trace[b] + gen_trace_last[s][b]
                            if len(gen_trace_last[s])
                            else last_step_trace[b]
                        )
                        for b in range(len(last_step_trace))
                    ]
                    for s in range(len(gen_trace_last))
                ]
            else:
                generation_trace = gen_trace_last
        except:
            generation_trace = [
                tokenizer.batch_decode(x[:, Lp:], skip_special_tokens=True)
            ]

        # Ensure gen_trace_last is always defined
        if gen_trace_last is None:
            gen_trace_last = generation_trace

        # Full trace: cumulative trace across all non-empty blocks
        # [block1 trace] + [block1_last + block2 trace] + [block1_last + block2_last + block3 trace]
        full_generation_trace = []
        prefix = []
        for block_i, block_trace in enumerate(gen_trace):
            # Skip empty blocks (blocks with 0 or 1 step, meaning no meaningful generation)
            if len(block_trace) <= 1:
                continue
            # Initialize prefix with empty strings if this is the first non-empty block
            if not prefix and block_trace[0]:
                prefix = [""] * len(block_trace[0])
            # For each step in current block, prepend the prefix (final states of all previous blocks)
            for step_trace in block_trace:
                # Skip empty steps
                if not step_trace:
                    continue
                step_with_prefix = [
                    (prefix[b] + step_trace[b] if prefix[b] else step_trace[b])
                    for b in range(len(step_trace))
                ]
                full_generation_trace.append(step_with_prefix)
            # Update prefix to include the final state of this block
            if block_trace[-1]:
                prefix = [
                    prefix[b] + block_trace[-1][b] if prefix[b] else block_trace[-1][b]
                    for b in range(len(block_trace[-1]))
                ]

        # Pad and reshape gen_trace_logits/entropy: list of blocks -> (B, num_blocks, max_steps, block_length)
        # Pad with -inf for easy filtering later (matches generate_with_dual_cache)
        gen_trace_logits_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.stack(inner, dim=0) for inner in gen_trace_logits],
            batch_first=True,
            padding_value=float('-inf'),
        )  # (num_blocks, max_steps, B, block_length)
        gen_trace_logits_tensor = gen_trace_logits_padded.permute(2, 0, 1, 3)  # (B, num_blocks, max_steps, block_length)

        gen_trace_entropy_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.stack(inner, dim=0) for inner in gen_trace_entropy],
            batch_first=True,
            padding_value=float('-inf'),
        )  # (num_blocks, max_steps, B, block_length)
        gen_trace_entropy_tensor = gen_trace_entropy_padded.permute(2, 0, 1, 3)  # (B, num_blocks, max_steps, block_length)

        gen_trace_argmax_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.stack(inner, dim=0) for inner in gen_trace_argmax],
            batch_first=True,
            padding_value=-1,
        )  # (num_blocks, max_steps, B, block_length)
        gen_trace_argmax_tensor = gen_trace_argmax_padded.permute(2, 0, 1, 3)  # (B, num_blocks, max_steps, block_length)

        gen_trace_top2gap_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.stack(inner, dim=0) for inner in gen_trace_top2gap],
            batch_first=True,
            padding_value=float('-inf'),
        )
        gen_trace_top2gap_tensor = gen_trace_top2gap_padded.permute(2, 0, 1, 3)

        remask_stats = {
            "generation_trace": generation_trace,  # original: last block only
            "full_generation_trace": full_generation_trace,  # new: full cumulative trace
            "last_block_generation_trace": gen_trace_last,  # trace of last block only (for GSM8K-style answers)
            "block_generation_traces": gen_trace,  # per-block traces without concatenation
            "remask_count": mask_trace,  # number of times each token was remasked
            "first_remask_step": first_transfer_step,  # step at which each token was first transferred (-1 if never transferred)
            "gen_trace_logits": gen_trace_logits_tensor,
            "gen_trace_entropy": gen_trace_entropy_tensor,
            "gen_trace_argmax": gen_trace_argmax_tensor,
            "gen_trace_top2gap": gen_trace_top2gap_tensor,
            "confidence_at_commit": confidence_at_commit,
            "mask_fraction_at_commit": mask_fraction_at_commit,
        }
        return x, nfe, remask_stats
    else:
        return x, nfe


def apply_shift_logits(logits):
    """Shift logits right by 1 for Dream-style models that predict position i+1 from hidden state i."""
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


@torch.no_grad()
def generate_with_prefix_cache(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
    factor=None,
    shift_logits=False,
    tokenizer=None,
):
    """Semi-autoregressive generation with prefix KV-cache.

    Requires a model whose forward() supports ``use_cache=True`` and returns
    ``past_key_values`` as ``List[Tuple[K, V]]``.  Works with Fast-dLLM's
    DreamModel (cache shape ``(B, seq, hidden)``) and LLaDA 1.5 (cache shape
    ``(B, heads, seq, head_dim)``).
    """
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    x = torch.full(
        (B, Lp + gen_length), mask_id, dtype=torch.long
    ).to(model.device)
    x[:, :Lp] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    # UQ statistics
    remask_count = torch.zeros((B, Lp + gen_length), dtype=torch.long, device=model.device)
    first_transfer_step = torch.full((B, Lp + gen_length), -1, dtype=torch.long, device=model.device)
    confidence_at_commit = torch.full((B, Lp + gen_length), float('-inf'), device=model.device)
    mask_fraction_at_commit = torch.full((B, Lp + gen_length), float('-inf'), device=model.device)
    gen_trace = []
    mask_trace = []
    gen_trace_logits = []
    gen_trace_entropy = []
    gen_trace_argmax = []
    gen_trace_top2gap = []

    nfe = 0
    global_step = 0

    for num_block in range(num_blocks):
        gen_trace.append([])
        mask_trace.append([])
        gen_trace_logits.append([])
        gen_trace_entropy.append([])
        gen_trace_argmax.append([])
        gen_trace_top2gap.append([])
        s = Lp + num_block * block_length
        e = s + block_length

        block_mask_index = x[:, s:e] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # Full forward with cache (warm-up)
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        logits = output.logits
        if logits is None:
            # Diagnostic dump for the CIAI-only sbatch bug where the cached
            # forward returns logits=None despite the bare model in interactive
            # probes returning a populated tensor. Inspects model.forward (which
            # may be wrapped as functools.partial by accelerate's device hooks
            # or similar) and dumps full state, then falls back to a
            # use_cache=False forward, which always returns logits in probes.
            import sys, functools
            available = list(getattr(output, "__dataclass_fields__", {}).keys()) or dir(output)

            # forward inspection: real method vs partial vs other
            fwd = model.forward
            if isinstance(fwd, functools.partial):
                inner = getattr(fwd, "func", None)
                inner_repr = (
                    f"partial(func={getattr(inner, '__qualname__', repr(inner))}, "
                    f"args={fwd.args!r}, kwargs_keys={list((fwd.keywords or {}).keys())!r})"
                )
                inner_file = getattr(getattr(inner, "__code__", None), "co_filename", "N/A")
            else:
                inner_repr = repr(fwd)
                inner_file = getattr(getattr(fwd, "__code__", None), "co_filename", "N/A")

            # accelerate hook detection
            hf_hook = getattr(model, "_hf_hook", None)
            hook_repr = type(hf_hook).__name__ if hf_hook is not None else "None"

            sys.stderr.write(
                f"[DIAG generate.py:481] logits=None at num_block={num_block} s={s} e={e}\n"
                f"  output type={type(output).__module__}.{type(output).__name__}\n"
                f"  output is tuple? {isinstance(output, tuple)}\n"
                f"  output fields/attrs={available[:25]}\n"
                f"  past_key_values type={type(past_key_values).__name__}; "
                f"len={len(past_key_values) if past_key_values is not None else 'None'}\n"
                f"  x.shape={tuple(x.shape)} dtype={x.dtype} device={x.device}\n"
                f"  model class={type(model).__module__}.{type(model).__name__}\n"
                f"  model.forward repr={inner_repr}\n"
                f"  model.forward source file={inner_file}\n"
                f"  model._hf_hook={hook_repr}\n"
                f"  num_logits_to_keep config={getattr(model.config, 'num_logits_to_keep', 'MISSING')}\n"
                f"  use_cache config={getattr(model.config, 'use_cache', 'MISSING')}\n"
                f"  retrying with use_cache=False...\n"
            )
            sys.stderr.flush()
            fallback = model(x, use_cache=False)
            sys.stderr.write(
                f"[DIAG generate.py:481] fallback logits is None? {fallback.logits is None}; "
                f"shape={tuple(fallback.logits.shape) if fallback.logits is not None else 'N/A'}\n"
            )
            sys.stderr.flush()
            if fallback.logits is None:
                raise RuntimeError(
                    f"Dream forward returned logits=None even with use_cache=False; "
                    f"output type {type(output).__name__}, fields {available[:25]}"
                )
            logits = fallback.logits
            past_key_values = fallback.past_key_values  # may be None for use_cache=False
            if past_key_values is None:
                raise RuntimeError(
                    "generate_with_prefix_cache: primary forward returned logits=None and "
                    "the use_cache=False fallback produced no past_key_values, so the KV "
                    "cache cannot be maintained (it is indexed below). Load Dream without an "
                    "accelerate device_map (see llada/load_dream_cached.py) so the cached "
                    "forward returns logits directly."
                )
        if shift_logits:
            raw_logit_before_block = logits[:, s - 1:s, :].clone()
            logits = apply_shift_logits(logits)
        nfe += 1

        # Step 0: transfer only the first block token (like Fast-dLLM).
        # Transferring many tokens at step 0 degrades cache quality because
        # the cached prefix KV was computed with all-MASK block tokens.
        x0_full = logits.argmax(-1)
        transfer_index = torch.zeros_like(x, dtype=torch.bool)
        transfer_index[:, s] = True  # only first token of block

        # Collect UQ traces for step 0
        if tokenizer is not None:
            logits_block = logits[:, s:e]
            gen_trace_logits[-1].append(logits_block.log_softmax(-1).max(-1).values)
            probs_block = logits_block.softmax(-1)
            gen_trace_entropy[-1].append(-(probs_block * probs_block.log()).sum(-1))
            gen_trace_argmax[-1].append(logits_block.argmax(-1))
            top2 = probs_block.topk(2, dim=-1).values
            gen_trace_top2gap[-1].append(top2[:, :, 0] - top2[:, :, 1])
            gen_trace[-1].append(tokenizer.batch_decode(x0_full[:, s:e], skip_special_tokens=True))
            mask_trace[-1].append(x[:, s:e] == mask_id)

        x[transfer_index] = x0_full[transfer_index]

        # Update commitment stats
        gen_mask = torch.zeros_like(transfer_index)
        gen_mask[:, Lp:] = True
        newly_committed = transfer_index & (first_transfer_step == -1) & gen_mask
        remask_count = remask_count + transfer_index.long()
        first_transfer_step = torch.where(newly_committed, global_step, first_transfer_step)
        if tokenizer is not None:
            step_logprobs = logits.log_softmax(-1).max(-1).values
            confidence_at_commit = torch.where(newly_committed, step_logprobs, confidence_at_commit)
            current_mask_frac = (x == mask_id).float().sum(-1, keepdim=True) / x.shape[1]
            mask_fraction_at_commit = torch.where(
                newly_committed, current_mask_frac.expand_as(newly_committed), mask_fraction_at_commit,
            )
        global_step += 1

        # Truncate cache to prompt + completed blocks only.
        # Detect cache format: Fast-dLLM stores (B, seq, hidden) — 3D,
        # LLaDA 1.5 stores (B, heads, seq, head_dim) — 4D.
        sample_kv = past_key_values[0][0]
        seq_dim = 1 if sample_kv.dim() == 3 else 2

        new_past_key_values = []
        for layer_i in range(len(past_key_values)):
            new_past_key_values.append(tuple(
                kv.narrow(seq_dim, 0, s) for kv in past_key_values[layer_i]
            ))
        past_key_values = new_past_key_values

        # Refinement steps with cached prefix
        i = 1
        while True:
            if (x[:, s:e] == mask_id).sum() == 0:
                break
            nfe += 1

            # Forward only suffix (current_block_start:) with cached prefix
            out_blk = model(
                x[:, s:],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits_suffix = out_blk.logits
            if shift_logits:
                logits_suffix = torch.cat([raw_logit_before_block, logits_suffix[:, :-1]], dim=1)

            mask_index = x[:, s:] == mask_id
            mask_index[:, block_length:] = 0

            if factor is None:
                x0, transfer_index = get_transfer_index(
                    logits_suffix, temperature, remasking, mask_index, x[:, s:],
                    num_transfer_tokens[:, i] if threshold is None else None, threshold,
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(
                    logits_suffix, temperature, remasking, mask_index, x[:, s:], None, factor,
                )

            # Collect UQ traces — logits_suffix[:, :block_length] corresponds to the current block
            if tokenizer is not None:
                logits_blk = logits_suffix[:, :block_length]
                gen_trace_logits[-1].append(logits_blk.log_softmax(-1).max(-1).values)
                probs_blk = logits_blk.softmax(-1)
                gen_trace_entropy[-1].append(-(probs_blk * probs_blk.log()).sum(-1))
                gen_trace_argmax[-1].append(logits_blk.argmax(-1))
                top2_blk = probs_blk.topk(2, dim=-1).values
                gen_trace_top2gap[-1].append(top2_blk[:, :, 0] - top2_blk[:, :, 1])
                gen_trace[-1].append(tokenizer.batch_decode(x0[:, :block_length], skip_special_tokens=True))
                mask_trace[-1].append(x[:, s:e] == mask_id)

            x[:, s:][transfer_index] = x0[transfer_index]

            # Update commitment stats
            full_transfer_index = torch.zeros_like(x, dtype=torch.bool)
            full_transfer_index[:, s:][transfer_index] = True
            remask_count = remask_count + full_transfer_index.long()
            newly_committed = full_transfer_index & (first_transfer_step == -1) & gen_mask
            first_transfer_step = torch.where(newly_committed, global_step, first_transfer_step)
            if tokenizer is not None:
                full_logprobs = torch.full_like(confidence_at_commit, float('-inf'))
                full_logprobs[:, s:s + logits_suffix.shape[1]] = logits_suffix.log_softmax(-1).max(-1).values
                confidence_at_commit = torch.where(newly_committed, full_logprobs, confidence_at_commit)
                current_mask_frac = (x == mask_id).float().sum(-1, keepdim=True) / x.shape[1]
                mask_fraction_at_commit = torch.where(
                    newly_committed, current_mask_frac.expand_as(newly_committed), mask_fraction_at_commit,
                )
            global_step += 1
            i += 1

    if tokenizer is not None:
        # Build remask_stats identical to generate()
        mask_trace = torch.stack(
            [torch.stack(inner, dim=0).int().sum(0) for inner in mask_trace], dim=0
        )
        mask_trace = mask_trace.permute(1, 0, 2)
        mask_trace = mask_trace.reshape(mask_trace.shape[0], -1)

        gen_trace_last = None
        try:
            from_end_index, gen_trace_last = next(
                (i, x) for i, x in enumerate(reversed(gen_trace)) if len(x) > 1
            )
            last_block = len(gen_trace) - from_end_index - 1
            last_step_trace = [
                "".join(gen_trace[block_i][-1][batch_i] for block_i in range(last_block))
                for batch_i in range(len(gen_trace[0][1]))
            ]
            if len(last_step_trace):
                generation_trace = [
                    [
                        (last_step_trace[b] + gen_trace_last[s_][b] if len(gen_trace_last[s_]) else last_step_trace[b])
                        for b in range(len(last_step_trace))
                    ]
                    for s_ in range(len(gen_trace_last))
                ]
            else:
                generation_trace = gen_trace_last
        except:
            generation_trace = [tokenizer.batch_decode(x[:, Lp:], skip_special_tokens=True)]

        if gen_trace_last is None:
            gen_trace_last = generation_trace

        full_generation_trace = []
        prefix = []
        for block_i, block_trace in enumerate(gen_trace):
            if len(block_trace) <= 1:
                continue
            if not prefix and block_trace[0]:
                prefix = [""] * len(block_trace[0])
            for step_trace in block_trace:
                if not step_trace:
                    continue
                step_with_prefix = [
                    (prefix[b] + step_trace[b] if prefix[b] else step_trace[b])
                    for b in range(len(step_trace))
                ]
                full_generation_trace.append(step_with_prefix)
            if block_trace[-1]:
                prefix = [
                    prefix[b] + block_trace[-1][b] if prefix[b] else block_trace[-1][b]
                    for b in range(len(block_trace[-1]))
                ]

        gen_trace_logits_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.stack(inner, dim=0) for inner in gen_trace_logits],
            batch_first=True, padding_value=float('-inf'),
        )
        gen_trace_logits_tensor = gen_trace_logits_padded.permute(2, 0, 1, 3)

        gen_trace_entropy_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.stack(inner, dim=0) for inner in gen_trace_entropy],
            batch_first=True, padding_value=float('-inf'),
        )
        gen_trace_entropy_tensor = gen_trace_entropy_padded.permute(2, 0, 1, 3)

        gen_trace_argmax_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.stack(inner, dim=0) for inner in gen_trace_argmax],
            batch_first=True, padding_value=-1,
        )
        gen_trace_argmax_tensor = gen_trace_argmax_padded.permute(2, 0, 1, 3)

        gen_trace_top2gap_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.stack(inner, dim=0) for inner in gen_trace_top2gap],
            batch_first=True, padding_value=float('-inf'),
        )
        gen_trace_top2gap_tensor = gen_trace_top2gap_padded.permute(2, 0, 1, 3)

        remask_stats = {
            "generation_trace": generation_trace,
            "full_generation_trace": full_generation_trace,
            "last_block_generation_trace": gen_trace_last,
            "block_generation_traces": gen_trace,
            "remask_count": mask_trace,
            "first_remask_step": first_transfer_step,
            "gen_trace_logits": gen_trace_logits_tensor,
            "gen_trace_entropy": gen_trace_entropy_tensor,
            "gen_trace_argmax": gen_trace_argmax_tensor,
            "gen_trace_top2gap": gen_trace_top2gap_tensor,
            "confidence_at_commit": confidence_at_commit,
            "mask_fraction_at_commit": mask_fraction_at_commit,
        }
        return x, nfe, remask_stats
    else:
        return x, nfe


@torch.no_grad()
def generate_with_dual_cache(
    model,
    tokenizer,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
    factor=None,
    output_hidden_states=False,
    hs_layer=-1,
    output_all_hidden_states_trace=False,
):
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])  # Python int, not Tensor
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # x: (B, Lp + gen_length)
    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt

    # Statistics: track remasking count for each token position
    remask_count = torch.zeros(
        (B, Lp + gen_length), dtype=torch.long, device=model.device
    )
    # Track the step at which each token was first transferred (unmasked)
    first_transfer_step = torch.full(
        (B, Lp + gen_length), -1, dtype=torch.long, device=model.device
    )
    confidence_at_commit = torch.full(
        (B, Lp + gen_length), float('-inf'), device=model.device
    )
    mask_fraction_at_commit = torch.full(
        (B, Lp + gen_length), float('-inf'), device=model.device
    )
    gen_trace = []
    mask_trace = []
    gen_trace_logits = []
    gen_trace_entropy = []
    gen_trace_argmax = []
    gen_trace_top2gap = []
    hs_trace = []

    def _mean_pool_unmasked(state, token_ids):
        valid = (token_ids != mask_id).unsqueeze(-1).to(
            device=state.device, dtype=state.dtype
        )
        return (state * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    nfe = 0
    global_step = 0

    for nb in range(num_blocks):
        gen_trace.append([])
        mask_trace.append([])
        gen_trace_logits.append([])
        gen_trace_entropy.append([])
        gen_trace_argmax.append([])
        gen_trace_top2gap.append([])
        hs_trace.append([])
        s = Lp + nb * block_length
        e = s + block_length

        # Masks/indices for the current block
        block_mask_index = x[:, s:e] == mask_id  # (B, block_length)

        num_transfer_tokens = get_num_transfer_tokens(
            block_mask_index, steps_per_block
        )  # (B, steps_per_block)

        # 1) Warm KV-cache on the full prefix once per block
        out_full = model(x, use_cache=True, output_hidden_states=output_hidden_states)
        past_key_values = out_full.past_key_values
        if output_hidden_states:
            if output_all_hidden_states_trace:
                hs_trace[-1].append(
                    torch.stack(
                        [
                            _mean_pool_unmasked(
                                state[:, s:e], x[:, s:e]
                            ).detach().cpu()
                            for state in out_full.hidden_states
                        ],
                        dim=1,
                    )
                )
            else:
                hs_trace[-1].append(
                    out_full.hidden_states[hs_layer][:, s:e].detach().cpu()
                )
        nfe += 1

        # Build a replace_position tensor indicating the block range (static slice)
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True  # boolean mask (not a dynamic slice bound)

        # Step 0: do an initial transfer on the full logits
        global_mask_index = x == mask_id
        # Do not touch beyond current block in this phase
        global_mask_index[:, e:] = False

        if factor is None:
            quota0 = (
                None if threshold is not None else num_transfer_tokens[:, 0]
            )  # (B,)
            x0, transfer_index = get_transfer_index(
                out_full.logits,
                temperature,
                remasking,
                global_mask_index,
                x,
                quota0,
                threshold,
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                out_full.logits,
                temperature,
                remasking,
                global_mask_index,
                x,
                None,
                factor,
            )
        gen_trace[-1].append(
            tokenizer.batch_decode(x0[:, s:e], skip_special_tokens=True)
        )
        mask_trace[-1].append(block_mask_index)

        # Track logits and entropy
        # For the generated block only (s:e)
        logits_block = out_full.logits[:, s:e]  # (B, block_length, vocab_size)
        gen_trace_logits[-1].append(
            logits_block.log_softmax(-1).max(-1).values
        )  # (B, block_length)
        # Entropy: p * log(p) -> mean over tokens
        probs_block = logits_block.softmax(-1)  # (B, block_length, vocab_size)
        entropy_block = -(probs_block * probs_block.log()).sum(-1)  # (B, block_length)
        gen_trace_entropy[-1].append(entropy_block)
        gen_trace_argmax[-1].append(logits_block.argmax(-1))
        top2_block = probs_block.topk(2, dim=-1).values
        gen_trace_top2gap[-1].append(top2_block[:, :, 0] - top2_block[:, :, 1])
        # In-place update via torch.where (no tensor-slice assignment with mask)
        x = torch.where(transfer_index, x0, x)

        # Update remasking statistics for step 0
        remask_count = remask_count + transfer_index.long()
        newly_committed = transfer_index & (first_transfer_step == -1)
        first_transfer_step = torch.where(
            newly_committed, global_step, first_transfer_step,
        )
        step_logprobs = out_full.logits.log_softmax(-1).max(-1).values
        confidence_at_commit = torch.where(
            newly_committed, step_logprobs, confidence_at_commit,
        )
        current_mask_frac = (x == mask_id).float().sum(-1, keepdim=True) / x.shape[1]
        mask_fraction_at_commit = torch.where(
            newly_committed,
            current_mask_frac.expand_as(newly_committed),
            mask_fraction_at_commit,
        )
        global_step += 1

        # 2) Semi-autoregressive refinement, fixed number of steps (graph-friendly)
        #    Each iteration runs on the current block with KV-cache and replace_position
        for i in range(1, steps_per_block):
            # Evaluate logits only for current block with cache
            if (x[:, s:e] == mask_id).sum() == 0:
                break
            out_blk = model(
                x[:, s:e],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position,
                output_hidden_states=output_hidden_states,
            )
            logits_blk = out_blk.logits  # shape expected by get_transfer_index*
            if output_hidden_states:
                if output_all_hidden_states_trace:
                    hs_trace[-1].append(
                        torch.stack(
                            [
                                _mean_pool_unmasked(
                                    state[:, -block_length:], x[:, s:e]
                                ).detach().cpu()
                                for state in out_blk.hidden_states
                            ],
                            dim=1,
                        )
                    )
                else:
                    hs_trace[-1].append(
                        out_blk.hidden_states[hs_layer][
                            :, -block_length:
                        ].detach().cpu()
                    )

            # Mask and quota for this step (all tensor ops)
            mask_blk = x[:, s:e] == mask_id  # (B, block_length)

            if factor is None:
                quota_i = (
                    None if threshold is not None else num_transfer_tokens[:, i]
                )  # (B,)
                x0_blk, transfer_idx_blk = get_transfer_index(
                    logits_blk,
                    temperature,
                    remasking,
                    mask_blk,
                    x[:, s:e],
                    quota_i,
                    threshold,
                )
            else:
                x0_blk, transfer_idx_blk = get_transfer_index_dynamic(
                    logits_blk,
                    temperature,
                    remasking,
                    mask_blk,
                    x[:, s:e],
                    None,
                    factor,
                )

            mask_trace[-1].append(mask_blk)
            gen_trace[-1].append(
                tokenizer.batch_decode(x0_blk, skip_special_tokens=True)
            )

            # Track logits and entropy for refinement steps
            gen_trace_logits[-1].append(
                logits_blk.log_softmax(-1).max(-1).values
            )  # (B, block_length)
            probs_blk = logits_blk.softmax(-1)  # (B, block_length, vocab_size)
            entropy_blk = -(probs_blk * probs_blk.log()).sum(-1)  # (B, block_length)
            gen_trace_entropy[-1].append(entropy_blk)
            gen_trace_argmax[-1].append(logits_blk.argmax(-1))
            top2_blk = probs_blk.topk(2, dim=-1).values
            gen_trace_top2gap[-1].append(top2_blk[:, :, 0] - top2_blk[:, :, 1])
            # Merge back into x[:, s:e] using torch.where (no masked slice assignment)
            blk_old = x[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)  # static concatenation

            # Update remasking statistics for refinement steps
            # Create full-size transfer index for the current block
            full_transfer_index = torch.zeros_like(x, dtype=torch.bool)
            full_transfer_index[:, s:e] = transfer_idx_blk
            remask_count = remask_count + full_transfer_index.long()
            # Track first transfer step and commitment stats
            newly_committed_blk = full_transfer_index & (first_transfer_step == -1)
            first_transfer_step = torch.where(
                newly_committed_blk, global_step, first_transfer_step,
            )
            # Confidence at commitment: expand block logits to full sequence size
            full_logprobs = torch.full_like(confidence_at_commit, float('-inf'))
            full_logprobs[:, s:e] = logits_blk.log_softmax(-1).max(-1).values
            confidence_at_commit = torch.where(
                newly_committed_blk, full_logprobs, confidence_at_commit,
            )
            current_mask_frac = (x == mask_id).float().sum(-1, keepdim=True) / x.shape[1]
            mask_fraction_at_commit = torch.where(
                newly_committed_blk,
                current_mask_frac.expand_as(newly_committed_blk),
                mask_fraction_at_commit,
            )
            global_step += 1

            nfe += 1

    if output_hidden_states:
        out_final = model(x, use_cache=False, output_hidden_states=True)
        for block_idx in range(num_blocks):
            s = Lp + block_idx * block_length
            e = s + block_length
            if output_all_hidden_states_trace:
                hs_trace[block_idx].append(
                    torch.stack(
                        [
                            _mean_pool_unmasked(
                                state[:, s:e], x[:, s:e]
                            ).detach().cpu()
                            for state in out_final.hidden_states
                        ],
                        dim=1,
                    )
                )
            else:
                hs_trace[block_idx].append(
                    out_final.hidden_states[hs_layer][:, s:e].detach().cpu()
                )
        del out_final
        nfe += 1

    mask_trace = torch.stack(
        [torch.stack(inner, dim=0).int().sum(0) for inner in mask_trace], dim=0
    )
    mask_trace = mask_trace.permute(1, 0, 2)
    mask_trace = mask_trace.reshape(mask_trace.shape[0], -1)

    # Original behavior: trace only for the last non-empty block
    gen_trace_last = None  # Initialize before try/except
    try:
        from_end_index, gen_trace_last = next(
            (i, x) for i, x in enumerate(reversed(gen_trace)) if len(x) > 1
        )
        last_block = len(gen_trace) - from_end_index - 1
        last_step_trace = [
            "".join(gen_trace[block_i][-1][batch_i] for block_i in range(last_block))
            for batch_i in range(len(gen_trace[0][1]))
        ]
        if len(last_step_trace):
            generation_trace = [
                [
                    (
                        last_step_trace[b] + gen_trace_last[s][b]
                        if len(gen_trace_last[s])
                        else last_step_trace[b]
                    )
                    for b in range(len(last_step_trace))
                ]
                for s in range(len(gen_trace_last))
            ]
        else:
            generation_trace = gen_trace_last
    except:
        generation_trace = [tokenizer.batch_decode(x[:, Lp:], skip_special_tokens=True)]

    # Ensure gen_trace_last is always defined
    if gen_trace_last is None:
        gen_trace_last = generation_trace

    # Full trace: cumulative trace across all non-empty blocks
    # [block1 trace] + [block1_last + block2 trace] + [block1_last + block2_last + block3 trace]
    full_generation_trace = []
    prefix = []
    for block_i, block_trace in enumerate(gen_trace):
        # Skip empty blocks (blocks with 0 or 1 step, meaning no meaningful generation)
        if len(block_trace) < 1:
            continue
        # Initialize prefix with empty strings if this is the first non-empty block
        if not prefix and block_trace[0]:
            prefix = [""] * len(block_trace[0])
        # For each step in current block, prepend the prefix (final states of all previous blocks)
        for step_trace in block_trace:
            # Skip empty steps
            if not step_trace:
                continue
            step_with_prefix = [
                (prefix[b] + step_trace[b] if prefix[b] else step_trace[b])
                for b in range(len(step_trace))
            ]
            full_generation_trace.append(step_with_prefix)
        # Update prefix to include the final state of this block
        if block_trace[-1]:
            prefix = [
                prefix[b] + block_trace[-1][b] if prefix[b] else block_trace[-1][b]
                for b in range(len(block_trace[-1]))
            ]

    # Convert trace logits and entropy to proper format
    # Use pad_sequence to handle variable number of steps per block
    # Pad with -inf for easy filtering later
    gen_trace_logits_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.stack(inner, dim=0) for inner in gen_trace_logits],
        batch_first=True,
        padding_value=float('-inf')
    )  # (num_blocks, max_steps, B, block_length)
    gen_trace_logits_tensor = gen_trace_logits_padded.permute(2, 0, 1, 3)  # (B, num_blocks, max_steps, block_length)

    gen_trace_entropy_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.stack(inner, dim=0) for inner in gen_trace_entropy],
        batch_first=True,
        padding_value=float('-inf')
    )  # (num_blocks, max_steps, B, block_length)
    gen_trace_entropy_tensor = gen_trace_entropy_padded.permute(2, 0, 1, 3)  # (B, num_blocks, max_steps, block_length)

    gen_trace_argmax_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.stack(inner, dim=0) for inner in gen_trace_argmax],
        batch_first=True,
        padding_value=-1,
    )  # (num_blocks, max_steps, B, block_length)
    gen_trace_argmax_tensor = gen_trace_argmax_padded.permute(2, 0, 1, 3)  # (B, num_blocks, max_steps, block_length)

    gen_trace_top2gap_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.stack(inner, dim=0) for inner in gen_trace_top2gap],
        batch_first=True,
        padding_value=float('-inf'),
    )
    gen_trace_top2gap_tensor = gen_trace_top2gap_padded.permute(2, 0, 1, 3)

    hidden_states_trace = None
    hidden_states_trace_mask = None

    if output_hidden_states:
        target_steps = (
            max(len(block) for block in hs_trace)
            if output_all_hidden_states_trace
            else steps_per_block + 1
        )
        block_traces = []
        block_masks = []

        for block in hs_trace:
            block_trace = torch.stack(block, dim=1)  # [B, T_block, L_block, H]
            valid_steps = block_trace.shape[1]

            block_trace = F.pad(
                block_trace,
                (0, 0, 0, 0, 0, target_steps - valid_steps),
            )

            block_mask = (
                torch.arange(target_steps).unsqueeze(0) < valid_steps
            ).expand(B, -1)

            block_traces.append(block_trace)
            block_masks.append(block_mask)

        hidden_states_trace = torch.stack(block_traces, dim=1)
        hidden_states_trace_mask = torch.stack(block_masks, dim=1)

    remask_stats = {
        "generation_trace": generation_trace,  # original: last block only
        "full_generation_trace": full_generation_trace,  # new: full cumulative trace
        "last_block_generation_trace": gen_trace_last,  # trace of last block only (for GSM8K-style answers)
        "block_generation_traces": gen_trace,  # per-block traces without concatenation
        "remask_count": mask_trace,  # number of times each token was remasked
        "first_remask_step": first_transfer_step,  # step at which each token was first transferred (-1 if never transferred)
        "gen_trace_logits": gen_trace_logits_tensor,  # max log-prob at each step
        "gen_trace_entropy": gen_trace_entropy_tensor,  # entropy at each step
        "gen_trace_argmax": gen_trace_argmax_tensor,  # argmax token ID at each step
        "gen_trace_top2gap": gen_trace_top2gap_tensor,  # top2 probability gap at each step
        "confidence_at_commit": confidence_at_commit,  # log-prob when each token was committed
        "mask_fraction_at_commit": mask_fraction_at_commit,  # fraction of sequence still masked at commitment
        "hidden_states_trace": hidden_states_trace if output_hidden_states else None,  # optional hidden states trace
        "hidden_states_trace_mask": hidden_states_trace_mask if output_hidden_states else None,  # optional hidden states trace mask
        "hidden_states_trace_block_length": block_length if output_hidden_states else None,
    }

    return x, nfe, remask_stats


def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,  # (B, L) bool
    x: torch.Tensor,  # (B, L) long
    num_transfer_tokens,  # (B,) or (B,1) long tensor, or None when threshold is used
    threshold: float = None,
):
    """
    Returns:
        x0: (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update this step
    """
    # 1) Sample proposal x0
    # Gumbel-noise for exploration; if temperature==0, add_gumbel_noise_optimized should no-op
    logits_with_noise = add_gumbel_noise_optimized(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L), long

    # 2) Confidence for chosen tokens (or random)
    if remasking == "low_confidence":
        # Use higher precision for softmax stability
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(
            -1
        )  # (B, L), float64
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)  # (B, L)
    else:
        raise NotImplementedError(remasking)

    # Only modify masked spots; keep others as original x and set their confidence to -inf
    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(
        torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype
    )
    confidence = torch.where(mask_index, x0_p, neg_inf)  # (B, L)

    # 3) Pick positions to transfer (vectorized)
    if threshold is not None:
        # Transfer all masked positions whose confidence >= threshold
        # (No top-k; purely threshold-based)
        transfer_index = mask_index & (confidence >= threshold)

        # at least one token is transferred "always unmask max c^i"
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)  # (B, 1)
        force_mask = torch.zeros_like(transfer_index).scatter_(
            1, max_conf_indices, True
        )

        # (Above Threshold) OR (Is Max Confidence)
        transfer_index = transfer_index | force_mask

        # Safety: do not unmask something that was not masked (consider fully unmasked rows)
        transfer_index = transfer_index & mask_index

        return x0, transfer_index

    # Else: per-row top-k with varying k (num_transfer_tokens), fully batched
    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    # Ensure shape (B,) long
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(
        dtype=torch.long, device=confidence.device
    )
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    # Sort confidences descending (masked positions are valid; others are -inf)
    # idx: (B, L) gives positions in original sequence sorted by confidence
    values, idx = torch.sort(confidence, dim=1, descending=True)

    B, L = confidence.shape
    # Build a mask that is True for the first k[b] columns in each row (sorted order)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)  # (B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)  # (B, L)
    select_sorted = cols < k_expanded  # (B, L) bool

    # Scatter the sorted True/False back to original column order
    # Use integer scatter then cast to bool (scatter_ on bool can be finicky across versions)
    transfer_int = torch.zeros(
        B, L, device=confidence.device, dtype=torch.int8
    )  # (B, L)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index  # ensure we never select unmasked

    return x0, transfer_index


def get_transfer_index_dynamic(
    logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1
):
    logits_with_noise = add_gumbel_noise_optimized(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
        )  # b, l
    elif remasking == "random":
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    for j in range(confidence.shape[0]):
        num_tokens = int(num_transfer_tokens[j].item())
        if num_tokens == 0:
            continue

        ns = list(range(1, num_transfer_tokens[j] + 1))
        es = [factor / (n + 1) for n in ns]
        threshs = [1 - e for e in es]

        # at least one token is transferred
        threshs[0] = -1
        sorted_confidence = torch.sort(
            confidence[j][mask_index[j]], dim=-1, descending=True
        )[0]
        assert len(sorted_confidence) == len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i] < threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs) - 1:
            top_i += 1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index
