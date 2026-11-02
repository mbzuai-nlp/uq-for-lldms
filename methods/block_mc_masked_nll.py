import numpy as np
from typing import Dict
from lm_polygraph.estimators.estimator import Estimator


class BlockMCMaskedNLL(Estimator):
    """
    Estimates sequence-level uncertainty via Monte Carlo estimates of the
    block-wise masked negative log-likelihood (BlockMCMaskedNLL).

    For semi-autoregressive diffusion LLMs, the answer is split into blocks
    of `block_size` tokens.  The ELBO is evaluated block-by-block: each block
    is masked independently while all previous blocks remain as clean context,
    matching the SAR generation protocol exactly.

    For each example, the method:
      - Takes `input_texts` as prompts and `greedy_texts` as the model's
        greedy completions.
      - Computes an MC estimate of the masked language model likelihood of
        the completion given the prompt using `get_log_likelihood_block`.
      - Returns the negative log-likelihood as the uncertainty score, with
        higher values indicating higher uncertainty. Empty answers are
        assigned a large fixed uncertainty.
    """

    def __init__(self):
        super().__init__(["block_mc_masked_ll"], "sequence")

    def __str__(self):
        return "BlockMCMaskedNLL"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        block_mc_masked_ll = stats["block_mc_masked_ll"]
        return -np.array(block_mc_masked_ll)


class BlockMCMaskedNormalizedNLL(Estimator):
    """
    Normalized version of BlockMCMaskedNLL (per-token uncertainty).
    """

    def __init__(self):
        super().__init__(["block_mc_masked_normalized_ll"], "sequence")

    def __str__(self):
        return "BlockMCMaskedNormalizedNLL"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        block_mc_masked_normalized_ll = stats["block_mc_masked_normalized_ll"]
        return -np.array(block_mc_masked_normalized_ll)


def load_estimator(cfg: Dict) -> Estimator:
    """
    Factory function to create a `BlockMCMaskedNLL` estimator from a config dictionary.
    """
    return BlockMCMaskedNLL()


def load_normalized_estimator(cfg: Dict) -> Estimator:
    """
    Factory function to create a `BlockMCMaskedNormalizedNLL` estimator from a config dictionary.
    """
    return BlockMCMaskedNormalizedNLL()
