import numpy as np

from typing import List, Dict, Optional

from lm_polygraph.estimators.estimator import Estimator


class CocoaMCMaskedNLL(Estimator):
    def __init__(
        self,
    ):
        super().__init__(["greedy_sentence_similarity", "mc_masked_ll"], "sequence")

    def __str__(self):
        return "CocoaMCMaskedNLL"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_sentence_similarity = stats["greedy_sentence_similarity"]
        batch_lls = np.array(stats["mc_masked_ll"])

        enriched_metrics = []  # To store enriched metrics for each sample
        for greedy_ll, greedy_sentence_similarity in zip(
            batch_lls, batch_greedy_sentence_similarity
        ):
            # Compute probabilities (negative log-probs)
            prob = -greedy_ll

            # Compute row-wise average similarity, excluding self-similarity
            # Diagonal contains self-similarities
            avg_dissimilarity = np.mean(1 - greedy_sentence_similarity)

            enriched_metric = prob * avg_dissimilarity
            enriched_metrics.append(enriched_metric)

        return np.array(enriched_metrics)


def load_estimator(cfg: Dict) -> Estimator:
    """
    Factory function to create a `CocoaMCMaskedNLL` estimator from a config dictionary.
    """
    return CocoaMCMaskedNLL()
