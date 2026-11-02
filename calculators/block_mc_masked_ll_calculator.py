import torch
import numpy as np

from typing import Dict, List, Optional, Tuple, Union

from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from whitebox_model_dllm import WhiteboxModelDiffusionLLM
from .mc_masked_ll_calculator import is_refusion_model, prepare_mc_batch
from llada.generate import build_block_attention_mask, apply_shift_logits

import torch.nn.functional as F


def forward_process_block(
    batch,
    block_start: int,
    block_end: int,
    mask_id: int,
    seed=None,
):
    """
    LLaDA-style masking applied only within a single SAR block [block_start, block_end).

    The positions *outside* the block (prompt + previous clean blocks) are left
    untouched.  Only the l_k masked positions inside the current block contribute
    to the loss, and the weight is L_k / l_k where L_k = block_end - block_start.

    Args:
        batch:       (B, L) token ids — full sequence (prompt + all answer tokens).
        block_start: Absolute column index where the current block begins.
        block_end:   Absolute column index where the current block ends (exclusive).
        mask_id:     Token id of the  token.
        seed:        Optional random seed.

    Returns:
        noisy_batch: (B, L) with l_k positions in [block_start, block_end) masked.
        p_mask:      (B, L) mask-ratio tensor; non-zero only in the block, used as
                     the 1/t denominator in the ELBO (value = l_k / L_k per row).
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    b, l = batch.shape
    block_len = block_end - block_start  # L_k

    # Sample l_k uniformly in {1, …, L_k}, spread across the batch with the
    # same linspace trick as forward_process to reduce variance.
    k = torch.randint(1, block_len + 1, (), device=batch.device)
    x = torch.round(
        torch.linspace(
            float(k), k + (b - 1) * (block_len / b), steps=b, device=batch.device
        )
    ).long()
    x = ((x - 1) % block_len) + 1  # x[i] = l_k for row i; in {1, …, L_k}

    # Build a boolean mask of shape (B, L_k): row i has x[i] True entries.
    indices = torch.arange(block_len, device=batch.device).repeat(b, 1)
    is_mask_block = indices < x.unsqueeze(1)
    for i in range(b):
        is_mask_block[i] = is_mask_block[i][torch.randperm(block_len)]

    # Embed the block mask into the full sequence shape.
    is_mask = torch.zeros(b, l, dtype=torch.bool, device=batch.device)
    is_mask[:, block_start:block_end] = is_mask_block

    noisy_batch = torch.where(is_mask, mask_id, batch)

    # p_mask[i, j] = l_k_i / L_k  for positions j in the block, else 0.
    p_mask = torch.zeros(b, l, device=batch.device)
    ratio = (x.float() / block_len).unsqueeze(1).expand(b, block_len)  # (B, L_k)
    p_mask[:, block_start:block_end] = ratio

    return noisy_batch, p_mask


def get_logits(model, batch, prompt_index, cfg_scale, mask_id, position_ids=None):
    if cfg_scale > 0.0:
        assert len(prompt_index) == batch.shape[1]
        prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
        un_batch = batch.clone()
        un_batch[prompt_index] = mask_id
        batch = torch.cat([batch, un_batch])
        if position_ids is not None:
            position_ids = torch.cat([position_ids, position_ids])

    input = batch
    model_kwargs = {}
    if position_ids is not None:
        model_kwargs["position_ids"] = position_ids
    if getattr(model.config, "model_type", "") == "llada2_moe":
        model_kwargs["attention_mask"] = build_block_attention_mask(
            input.shape[0], input.shape[1], input.device
        )
    logits = model(input, **model_kwargs).logits

    if cfg_scale > 0.0:
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    return logits


@torch.no_grad()
def get_log_likelihood_block(
    model,
    prompt,
    answer,
    block_size: int,
    mc_num: int = 128,
    batch_size: int = 16,
    cfg_scale: float = 0.0,
    mask_id: int = 126336,
    seed: Optional[int] = None,
    shift_logits: bool = False,
):
    """
    Conditional log-likelihood upper bound for a semi-autoregressive (SAR) model.

    The answer is split into blocks of `block_size` tokens (the last block may
    be shorter).  The bound decomposes as:

        -log p(r | p)  ≤  Σ_k  ELBO_k

    where each ELBO_k is the LLaDA discrete-diffusion bound applied *only
    within block k*, with all previous blocks serving as clean context:

        ELBO_k  =  𝔼_{l_k}[ (L_k/l_k) · Σ_i 𝟙[masked] · (-log p_θ(B_k^i | p, B_<k, B_k^{l_k})) ]

    Args:
        model:      Mask predictor (SAR variant with bidirectional attention
                    inside each block and causal conditioning across blocks).
        prompt:     1-D token tensor of shape (l_p,).
        answer:     1-D token tensor of shape (l_r,).
        block_size: Number of tokens per SAR block.
        mc_num:     Monte Carlo samples *per block*.
        batch_size: Mini-batch size for parallelising MC samples.
        cfg_scale:  Classifier-free guidance scale (0 = disabled).
        mask_id:    Token id of .
        seed:       Optional base random seed (each block/MC step gets seed+offset).

    Returns:
        ll_unnorm (float):    Summed (unnormalised) log-likelihood lower bound.
        ll_per_token (float): ll_unnorm / total answer length.
    """
    prompt_len = len(prompt)
    answer_len = len(answer)

    # Full sequence on device, replicated to batch_size rows.
    seq_1d = torch.cat([prompt, answer]).to(model.device)          # (L,)

    # Truncate from left if exceeding max sequence length
    max_sequence_length = 2048
    if len(seq_1d) > max_sequence_length:
        seq_1d = seq_1d[-max_sequence_length:]  # Keep rightmost tokens
        # Update prompt_len to reflect truncation (for block positioning)
        prompt_len = max(0, prompt_len - (len(prompt) + len(answer) - max_sequence_length))
        torch.cuda.empty_cache()  # Clear cached memory after truncation

    seq = seq_1d.unsqueeze(0).expand(batch_size, -1).clone()       # (B, L)

    refusion = is_refusion_model(model)
    # Block boundaries in *absolute* sequence coordinates.
    blocks: List[Tuple[int, int]] = []
    pos = prompt_len
    while pos < prompt_len + answer_len:
        end = min(pos + block_size, prompt_len + answer_len)
        blocks.append((pos, end))
        pos = end

    total_loss = 0.0
    n_estimates = 0

    for block_idx, (blk_start, blk_end) in enumerate(blocks):
        blk_len = blk_end - blk_start  # L_k

        if blk_len == 1:
            # Single-token block: no masking needed, compute exact cross-entropy.
            with torch.inference_mode():
                logits = get_logits(model, seq[:1], None, cfg_scale, mask_id)
            # Dream: score token at blk_start with the logit at blk_start-1.
            # No-op when shift_logits=False.
            pos = max(0, blk_start - 1) if shift_logits else blk_start
            ce = F.cross_entropy(
                logits[:1, pos],                # (1, vocab)
                seq[:1, blk_start],             # (1,)
                reduction="none",
            )
            total_loss += ce.sum().item()
            n_estimates += 1
            continue

        block_loss_ = []
        n_mc_rounds = max(1, mc_num // batch_size)

        for mc_idx in range(n_mc_rounds):
            used_seed = None
            if seed is not None:
                # Unique seed per (block, mc_round) to avoid correlation.
                used_seed = seed + block_idx * 10_000 + mc_idx

            if refusion:
                block_seq = seq[:, :blk_end]
                block_prompt_index = (
                    torch.arange(blk_end, device=seq.device) < blk_start
                )
                (
                    perturbed_seq,
                    labels,
                    mask_index,
                    p_mask,
                    position_ids,
                ) = prepare_mc_batch(
                    model,
                    block_seq,
                    block_prompt_index,
                    mask_id,
                    seed=used_seed,
                )
            else:
                perturbed_seq, p_mask = forward_process_block(
                    seq,
                    block_start=blk_start,
                    block_end=blk_end,
                    mask_id=mask_id,
                    seed=used_seed,
                )
                mask_index = perturbed_seq == mask_id
                labels = seq
                position_ids = None

            with torch.inference_mode():
                logits = get_logits(
                    model, perturbed_seq, None, cfg_scale, mask_id, position_ids
                )

            # Index-based shift_logits (see mc_masked_ll_calculator.py).
            # When shift_logits=False, src_l_idx == l_idx, so this selects the
            # exact same rows as logits[mask_index] — byte-identical to main.
            b_idx, l_idx = torch.where(mask_index)
            src_l_idx = (l_idx - 1).clamp(min=0) if shift_logits else l_idx
            selected_logits = logits[b_idx, src_l_idx]

            # Cross-entropy at masked positions, weighted by L_k / l_k = 1 / p_mask.
            loss = (
                F.cross_entropy(
                    selected_logits,      # (N_masked, vocab)
                    labels[mask_index],   # (N_masked,)
                    reduction="none",
                )
                / p_mask[mask_index]     # divide by l_k/L_k  ≡  multiply by L_k/l_k
            )
            block_loss_.append((loss.sum() / batch_size).item())

        # Average over MC rounds; accumulate into total.
        total_loss += sum(block_loss_) / len(block_loss_)
        n_estimates += 1

        # Clear GPU cache after each block to prevent memory accumulation
        torch.cuda.empty_cache()

    # ReFusion decomposes the answer likelihood as a sum over conditional blocks.
    ll_unnorm = -total_loss if refusion else -total_loss / len(blocks)
    ll_per_token = ll_unnorm / answer_len
    return ll_unnorm, ll_per_token


class BlockMCMaskedLLCalculator(StatCalculator):
    """
    MC masked log-likelihood for **semi-autoregressive** diffusion LLMs.

    The answer is split into blocks of `block_size` tokens.  The ELBO is
    evaluated block-by-block: each block is masked independently while all
    previous blocks remain as clean context, matching the SAR generation
    protocol exactly.

    The total bound is:

        -log p(r | p)  ≤  Σ_k  ELBO_k

    where ELBO_k uses the LLaDA discrete-diffusion bound inside block k
    with weight L_k / l_k  (Eq. 6 of arXiv:2502.09992, applied per block).
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return ["block_mc_masked_ll"], ["greedy_texts"]

    def __init__(
        self,
        block_size: int = 32,
        mc_num: int = 16,
        seed: Optional[int] = 42,
        mc_batch_size: int = 16,
    ):
        """
        Args:
            block_size: Number of tokens per SAR block (must match the model's
                        generation block size).
            mc_num:     Monte Carlo samples per block.
            seed:       Base random seed for reproducibility.
        """
        super().__init__()
        self.block_size = block_size
        self.mc_num = mc_num
        self.seed = seed
        self.mc_batch_size = mc_batch_size

    def __call__(
        self,
        dependencies: Dict[str, np.ndarray],
        texts: List[str],
        model: Union[WhiteboxModelDiffusionLLM],
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Returns:
            Dict with:
                - 'block_mc_masked_ll':            unnormalised LL lower bound (sum over blocks).
                - 'block_mc_masked_normalized_ll': LL per answer token.
        """
        model = dependencies["model"]
        tokenizer = dependencies["tokenizer"]
        greedy_texts = dependencies["greedy_texts"]
        input_texts = dependencies["input_texts"]
        # dependencies is a dict, so getattr never finds "mask_id" and always
        # returned the 126336 default (LLaDA-1.5) even for LLaDA-2.1/MoE (156895).
        # DiffusionGreedyProbsCalculator stores the model's mask_id under this
        # key, so read it by key.
        mask_id = dependencies.get("mask_id", 126336)

        block_ll = []
        block_ll_norm = []

        for i in range(len(greedy_texts)):
            prompt = torch.tensor(
                model.tokenize([input_texts[i]])["input_ids"]
            ).to(model.device()).reshape(-1)

            answer = torch.tensor(
                tokenizer(greedy_texts[i])["input_ids"]
            ).to(model.device())

            if len(answer):
                ll_i, ll_norm_i = get_log_likelihood_block(
                    model.model,
                    prompt,
                    answer,
                    block_size=self.block_size,
                    mc_num=self.mc_num,
                    batch_size=self.mc_batch_size,
                    mask_id=mask_id,
                    seed=self.seed,
                    shift_logits=getattr(model, "shift_logits", False),
                )
                block_ll.append(ll_i)
                block_ll_norm.append(ll_norm_i)
            else:
                block_ll.append(1e3)
                block_ll_norm.append(1e3)

            # Clear GPU cache after each sample to prevent OOM across iterations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            "block_mc_masked_ll": np.array(block_ll),
            "block_mc_masked_normalized_ll": np.array(block_ll_norm),
        }
