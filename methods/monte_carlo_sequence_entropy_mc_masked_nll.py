import numpy as np

from typing import Dict

from lm_polygraph.estimators.estimator import Estimator


class MonteCarloSequenceEntropyMCMaskedNLL(Estimator):
    def __init__(self):
        """
        Estimates the sequence-level uncertainty of a language model following the method of
        "Predictive entropy" as provided in the paper https://arxiv.org/abs/2302.09664.
        Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

        This method calculates the generation entropy estimations using Monte-Carlo estimation.
        The number of samples is controlled by lm_polygraph.stat_calculators.sample.SamplingGenerationCalculator
        'samples_n' parameter.
        """
        super().__init__(["sample_mc_masked_ll"], "sequence")

    def __str__(self):
        return "MonteCarloSequenceEntropyMCMaskedNLL"

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
        logprobs = stats["sample_mc_masked_ll"]
        return np.array([-np.mean(lp) for lp in logprobs])


def load_estimator(cfg: Dict) -> Estimator:
    """
    Factory function to create a `MonteCarloSequenceEntropyMCNLL` estimator from a config dictionary.
    """
    return MonteCarloSequenceEntropyMCMaskedNLL()
