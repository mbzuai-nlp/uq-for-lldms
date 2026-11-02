import random
import torch
import numpy as np

from refusion.qwen3.modeling_qwen3_refusion import (
    forward_process as refusion_forward_process,
)

from typing import Dict, List, Tuple, Union

from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from whitebox_model_dllm import WhiteboxModelDiffusionLLM
from llada.generate import build_block_attention_mask, apply_shift_logits

import torch.nn.functional as F


def forward_process(batch, prompt_index, mask_id, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    b, l = batch.shape

    target_len = (l - prompt_index.sum()).item()
    k = torch.randint(1, target_len + 1, (), device=batch.device)

    x = torch.round(
        torch.linspace(
            float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device
        )
    ).long()
    x = ((x - 1) % target_len) + 1
    assert x.min() >= 1 and x.max() <= target_len

    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]

    is_mask = torch.cat(
        (
            torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device),
            is_mask,
        ),
        dim=1,
    )
    noisy_batch = torch.where(is_mask, mask_id, batch)

    # Return the masked batch and the mask ratio
    return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)


def is_refusion_model(model):
    return "refusion" in type(model).__module__.lower()


def prepare_mc_batch(model, batch, prompt_index, mask_id, seed=None):
    if not is_refusion_model(model):
        noisy_batch, p_mask = forward_process(
            batch, prompt_index, mask_id, seed=seed
        )
        return noisy_batch, batch, noisy_batch == mask_id, p_mask, None

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    prompt_lengths = torch.full(
        (batch.shape[0], 1),
        int(prompt_index.sum().item()),
        dtype=torch.long,
        device=batch.device,
    )
    noisy_batch, labels, mask_index, p_mask, _, position_ids = refusion_forward_process(
        batch, torch.ones_like(batch), prompt_lengths, mask_id
    )
    return noisy_batch, labels, mask_index, p_mask, position_ids


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
def get_log_likelihood(
    model,
    prompt,
    answer,
    mc_num=128,
    batch_size=16,
    cfg_scale=0.0,
    mask_id=126336,
    seed=None,
    shift_logits=False,
):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (l1).
        answer: A tensor of shape (l2).
        mc_num: Monte Carlo estimation times.
                As detailed in Appendix B.5. Since MMLU, CMMLU, and C-EVAL only require the likelihood of a single token, a
                single Monte Carlo estimate is sufficient for these benchmarks. For all other benchmarks, we find that 128
                Monte Carlo samples are adequate to produce stable results.
        batch_size: Mini batch size.
        cfg_scale: Unsupervised classifier-free guidance scale.
        mask_id: The toke id of [MASK] is 126336.
        seed: Optional; random seed for MC masking/reproducibility.
    """
    # Truncate prompt to fit within max sequence length
    max_sequence_length = 2048
    total_len = len(prompt) + len(answer)
    if total_len > max_sequence_length:
        excess = total_len - max_sequence_length
        prompt = prompt[excess:]  # truncate from left
        
    seq = torch.cat([prompt, answer])[None, :]
    seq = seq.repeat((batch_size, 1)).to(model.device)
    prompt_index = torch.arange(seq.shape[1], device=model.device) < len(prompt)

    loss_ = []
    if answer.shape[0] > 1:
        for idx in range(mc_num // batch_size):
            # Optionally use seed+idx for determinism across MC samples
            used_seed = seed + idx if seed is not None else None
            perturbed_seq, labels, mask_index, p_mask, position_ids = (
                prepare_mc_batch(
                    model, seq, prompt_index, mask_id, seed=used_seed
                )
            )
            with torch.inference_mode():
                logits = get_logits(
                    model, perturbed_seq, prompt_index, cfg_scale, mask_id, position_ids
                )
                # Dream: index-based shift — score masked position j with the
                # logit at max(0, j-1). No-op when shift_logits=False.
                b_idx, l_idx = torch.where(mask_index)
                src_l_idx = (l_idx - 1).clamp(min=0) if shift_logits else l_idx
                selected_logits = logits[b_idx, src_l_idx]
            loss = (
                F.cross_entropy(
                    selected_logits,
                    labels[mask_index],
                    reduction="none",
                )
                / p_mask[mask_index]
            )
            loss_unnorm = loss.sum() / batch_size
            loss_.append(loss_unnorm.item())
    else:
        with torch.inference_mode():
            logits = get_logits(model, seq[:1], prompt_index, cfg_scale, mask_id)
            if shift_logits:
                # Dream: logit at position i predicts token i+1, so score
                # answer positions [len(prompt), L) with logits [len(prompt)-1, L-1).
                start = max(0, len(prompt) - 1)
                sliced_logits = logits[:, start:-1].reshape(-1, logits.shape[-1])
            else:
                sliced_logits = logits[:, len(prompt):].reshape(-1, logits.shape[-1])
            loss = F.cross_entropy(
                sliced_logits,
                seq[:1, len(prompt) :].reshape(-1),
                reduction="none",
            )

        loss_unnorm = loss.sum()
        loss_.append(loss_unnorm.item())

    answer_len = len(answer)
    ll_unnorm = -sum(loss_) / len(loss_)
    ll_per_token = ll_unnorm / answer_len
    return ll_unnorm, ll_per_token


class MCMaskedLLCalculator(StatCalculator):
    """
    For Whitebox model (lm_polygraph.WhiteboxModel), at input texts batch calculates:
    * MC masked log likelihoods
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """
        return [
            "mc_masked_ll",
        ], ["greedy_texts"]

    def __init__(
        self, mc_num: int = 16, seed: int = 42, mc_batch_size: int = 16
    ):
        super().__init__()
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
        Calculates the MC masked negative log-likelihoods (MCMaskedNLL) for each text in the batch.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, containing model, tokenizer, greedy_texts, and input_texts.
            texts (List[str]): Input texts batch used for model generation (ignored here).
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with
                - 'mc_masked_ll' (List[float]): MC masked log likelihood for each item.
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

        mc_masked_ll = []
        mc_masked_normalized_ll = []
        for i in range(len(greedy_texts)):
            prompt = model.tokenize([input_texts[i]])["input_ids"].detach().clone().to(
                model.device()
            ).reshape(-1)
            answer = tokenizer(greedy_texts[i], return_tensors="pt")["input_ids"].detach().clone().to(
                model.device()
            ).reshape(-1)
            if len(answer):
                mc_masked_ll_i, mc_masked_normalized_ll_i = get_log_likelihood(
                    model.model,
                    prompt,
                    answer,
                    mc_num=self.mc_num,
                    batch_size=self.mc_batch_size,
                    mask_id=mask_id,
                    seed=self.seed if self.seed is not None else None,
                    shift_logits=getattr(model, "shift_logits", False),
                )
                mc_masked_ll.append(mc_masked_ll_i)
                mc_masked_normalized_ll.append(mc_masked_normalized_ll_i)
            else:
                mc_masked_ll.append(1e3)
                mc_masked_normalized_ll.append(1e3)

        result_dict = {
            "mc_masked_ll": np.array(mc_masked_ll),
            "mc_masked_normalized_ll": np.array(mc_masked_normalized_ll),
        }

        return result_dict
