import numpy as np

from typing import Dict

from lm_polygraph.estimators.estimator import Estimator


class MonteCarloNormalizedSequenceEntropyMCMaskedNLL(Estimator):
    def __init__(self):
        """
        Estimates the sequence-level uncertainty of a language model following the method of
        "Predictive entropy" as provided in the paper https://arxiv.org/abs/2302.09664.
        Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

        This method calculates the generation entropy estimations using Monte-Carlo estimation with length normalization.
        The number of samples is controlled by lm_polygraph.stat_calculators.sample.SamplingGenerationCalculator
        'samples_n' parameter.
        """
        super().__init__(["sample_mc_masked_normalized_ll"], "sequence")

    def __str__(self):
        return "MonteCarloNormalizedSequenceEntropyMCMaskedNLL"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates generation entropy with Monte-Carlo for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * 'sample_mc_masked_ll': array of arrays containing the mc masked log likelihoods
                  (negative log likelihoods) for each sequence, for each sample.
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        # logprobs = stats["sample_mc_masked_ll"]
        # tokens = stats["sample_tokens"]
        # return np.array(
        #     [
        #         -np.mean([lp_i / len(t_i) for lp_i, t_i in zip(lp, t) if len(t_i)])
        #         for lp, t in zip(logprobs, tokens)
        #     ]
        # )
        logprobs = stats["sample_mc_masked_normalized_ll"]
        return np.array([-np.mean(lp) for lp in logprobs])


def load_estimator(cfg: Dict) -> Estimator:
    """
    Factory function to create a `MonteCarloSequenceEntropyMCNLL` estimator from a config dictionary.
    """
    return MonteCarloNormalizedSequenceEntropyMCMaskedNLL()
