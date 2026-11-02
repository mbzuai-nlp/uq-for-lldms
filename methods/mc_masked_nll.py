import numpy as np
from typing import Dict
from lm_polygraph.estimators.estimator import Estimator


class MCMaskedNLL(Estimator):
    """
    Estimates sequence-level uncertainty via Monte Carlo estimates of the
    masked negative log-likelihood (MCMaskedNLL) as proposed in
    https://arxiv.org/abs/2502.09992.

    For each example, the method:
      - Takes `input_texts` as prompts and `greedy_texts` as the model's
        greedy completions.
      - Computes an MC estimate of the masked language model likelihood of
        the completion given the prompt using `get_log_likelihood`.
      - Returns the negative log-likelihood as the uncertainty score, with
        higher values indicating higher uncertainty. Empty answers are
        assigned a large fixed uncertainty.
    """

    def __init__(self):
        super().__init__(["mc_masked_ll"], "sequence")

    def __str__(self):
        return "MCMaskedNLL"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        mc_masked_ll = stats["mc_masked_ll"]
        return -np.array(mc_masked_ll)


def load_estimator(cfg: Dict) -> Estimator:
    """
    Factory function to create a `MCMaskedNLL` estimator from a config dictionary.
    """
    return MCMaskedNLL()
