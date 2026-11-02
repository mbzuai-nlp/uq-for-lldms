from typing import Dict, List

import numpy as np

from lm_polygraph.stat_calculators.attention import LookbackRatioCalculator
from lm_polygraph.utils.model import WhiteboxModel


class SafeLookbackRatioCalculator(LookbackRatioCalculator):
    """Project-local lookback ratio calculator with a stable denominator."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = eps

    def __call__(
        self,
        dependencies: Dict[str, np.ndarray],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        del texts, max_new_tokens

        attentions = dependencies["attention_raw"]
        cut_sequences = dependencies["greedy_tokens"]

        lookback_ratios = []
        if model.model_type != "vLLMCausalLM":
            for i in range(len(cut_sequences)):
                sequence_length = len(cut_sequences[i])
                for token_idx in range(sequence_length):
                    token_ratios = []
                    for layer in range(len(attentions[token_idx])):
                        for head in range(len(attentions[token_idx][layer][0])):
                            token_attention = attentions[token_idx][layer][0][head][0]
                            if token_idx:
                                attention_on_new = (
                                    token_attention[-token_idx:].mean().item()
                                )
                                attention_on_context = (
                                    token_attention[:-token_idx].mean().item()
                                )
                            else:
                                attention_on_new = 0.0
                                attention_on_context = token_attention.mean().item()

                            denominator = (
                                attention_on_new + attention_on_context + self.eps
                            )
                            token_ratios.append(attention_on_context / denominator)
                    lookback_ratios.append(token_ratios)

        return {"lookback_ratios": lookback_ratios}
