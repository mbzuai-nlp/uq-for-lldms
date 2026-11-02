import numpy as np
import torch

from lm_polygraph.estimators.estimator import Estimator

from methods.cocoa_mc_masked_nll_trace import CocoaMCMaskedNLLTrace


class DCoCoAHybrid(CocoaMCMaskedNLLTrace):
    def __init__(
        self,
        alpha: float = 0.5,
        k: int = 1,
        inf_method: str = "MCMaskedNLL",
        multiply_by_nfe: bool = False,
        full_trace: bool = False,
        block_only: bool = False,
        block_agg: str = "avg",
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if k < 0:
            raise ValueError("k must be non-negative")
        if k == 0 and alpha != 0.0:
            raise ValueError("k=0 requires alpha=0 and denotes the baseline sanity check")
        if inf_method not in {
            "MCMaskedNLL",
            "MCMaskedNormalizedNLL",
            "ConfidenceAtCommit",
        }:
            raise ValueError(f"Unsupported hybrid information method: {inf_method}")

        super().__init__(
            inf_method=inf_method,
            multiply_by_nfe=multiply_by_nfe,
            full_trace=full_trace,
            block_only=block_only,
            block_agg=block_agg,
        )
        self.alpha = alpha
        self.k = k
        if self.k:
            self.stats_dependencies.append("greedy_sentence_similarity")

    def __str__(self):
        base_name = super().__str__().removeprefix("Cocoa")
        return f"D-CoCoA-Hybrid-{base_name}(K={self.k},alpha={self.alpha:g})"

    def _information(self, stats):
        if self.inf_method == "ConfidenceAtCommit":
            information = []
            for confidence in stats["confidence_at_commit"]:
                finite = confidence[torch.isfinite(confidence)]
                information.append(-finite.mean().item() if len(finite) else 0.0)
            information = np.asarray(information)
        else:
            information = -np.asarray(stats[self.inf_stat])

        if self.multiply_by_nfe:
            information = information * np.asarray(stats["nfe"])
        return information

    def __call__(self, stats):
        baseline = super().__call__(stats)
        if self.k == 0:
            return baseline

        sample_similarities = stats["greedy_sentence_similarity"]
        # Inputs with fewer than k samples (empty samples are dropped by the sampling calculator)
        # use the samples available; with none, the consistency term is 0.
        sample_dissimilarity = np.asarray(
            [
                np.mean(1.0 - np.asarray(similarities[: self.k])) if len(similarities) else 0.0
                for similarities in sample_similarities
            ]
        )
        return (
            self.alpha * baseline
            + (1.0 - self.alpha)
            * self._information(stats)
            * sample_dissimilarity
        )


def load_estimator(cfg) -> Estimator:
    return DCoCoAHybrid(
        alpha=cfg.get("alpha", 0.5),
        k=cfg.get("k", 1),
        inf_method=cfg.get("inf_method", "MCMaskedNLL"),
        multiply_by_nfe=cfg.get("multiply_by_nfe", False),
        full_trace=cfg.get("full_trace", False),
        block_only=cfg.get("block_only", False),
        block_agg=cfg.get("block_agg", "avg"),
    )
