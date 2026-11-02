import numpy as np
from typing import Dict

from lm_polygraph.estimators.estimator import Estimator


class LUQTrace(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "LUQ: Long-text Uncertainty Quantification for LLMs" as provided in the paper https://aclanthology.org/2024.emnlp-main.299.pdf.
    This class implements a basic version of LUQ without incorporating sentence splitting or atomic claim decomposition.
    Additionally, this implementation utilizes the default NLI model provided by lm_polygraph.utils.deberta.

    This is a trace-based version that uses generation_trace instead of sample_texts.
    """

    def __init__(self):
        super().__init__(
            [
                "semantic_matrix_entail_logits_trace",
                "semantic_matrix_contra_logits_trace",
            ],
            "sequence",
        )

    def __str__(self):
        return "LUQTrace"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the LUQ score for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * matrix with the logits of "entailment" class from the NLI model in 'semantic_matrix_entail_logits_trace'
                * matrix with the logits of "contradiction" class from the NLI model in 'semantic_matrix_contra_logits_trace'
        Returns:
            np.ndarray: float LUQ score for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        entail_logits = stats["semantic_matrix_entail_logits_trace"]
        contra_logits = stats["semantic_matrix_contra_logits_trace"]

        luq = []
        for j in range(len(entail_logits)):
            sim_scores = np.exp(entail_logits[j]) / (
                np.exp(entail_logits[j]) + np.exp(contra_logits[j])
            )
            n = sim_scores.shape[-1]
            if n <= 1:
                # Single draft in the trace => no off-diagonal pairs, no measurable
                # disagreement; avoid dividing by (n - 1) = 0, which yields NaN.
                luq.append(0.0)
                continue
            sim_scores = (sim_scores.sum(axis=1) - sim_scores.diagonal()) / (n - 1)
            luq.append(1 - sim_scores.mean())

        return np.array(luq)


def load_estimator(cfg: Dict) -> Estimator:
    """
    Factory function to create a `LUQTrace` estimator from a config dictionary.
    """
    return LUQTrace()
