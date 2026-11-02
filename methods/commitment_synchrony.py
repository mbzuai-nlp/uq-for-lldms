import numpy as np
import torch
from typing import Dict
from lm_polygraph.estimators.estimator import Estimator


class CommitmentSynchrony(Estimator):
    """
    Estimates sequence-level uncertainty based on how synchronized
    token commitment events are across the generation.

    When the model is confident, many tokens commit at the same step
    ("spike" or "phase transition") — the model makes a clear collective
    decision. When uncertain, commits are spread across many steps —
    the model dithers, committing tokens one by one.

    This is a sequence-level structural signal with no autoregressive
    equivalent — AR models commit every token at every step.

    Variants:
    - "gini": Gini coefficient of the commit-step distribution.
      High Gini = concentrated (few spikes) = more certain.
      Low Gini = spread out = uncertain.
    - "peak_ratio": ratio of (max bin count / mean bin count) in the
      commit-step histogram. High ratio = clear spike = certain.
    - "entropy": entropy of the commit-step distribution. Low entropy
      = concentrated commits = certain. High entropy = spread = uncertain.
    """

    def __init__(self, variant: str = "entropy"):
        super().__init__(["first_remask_step"], "sequence")
        self.variant = variant

    def __str__(self):
        return f"CommitSynchrony_{self.variant}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        first_step = stats["first_remask_step"]  # (B, gen_length)

        B = first_step.shape[0]
        result = []

        for i in range(B):
            steps = np.asarray(first_step[i])
            # Filter out -1 (tokens that were never transferred, e.g. prompt tokens)
            valid = steps[steps >= 0]

            if len(valid) < 2:
                result.append(0.0)
                continue

            # Count how many tokens committed at each step
            unique_steps, counts = np.unique(valid, return_counts=True)
            counts_float = counts.astype(np.float64)

            # Number of unique steps used vs total tokens — core synchrony measure
            n_unique = len(unique_steps)
            n_tokens = len(valid)

            if self.variant == "entropy":
                # Entropy of commit-step distribution
                # High entropy = spread out = uncertain
                probs = counts_float / counts_float.sum()
                entropy = -np.sum(probs * np.log(probs + 1e-12))
                result.append(entropy)

            elif self.variant == "peak_ratio":
                # Fraction of tokens that committed in the busiest step
                # High ratio = synchronized spike = certain
                peak_frac = counts_float.max() / n_tokens
                result.append(peak_frac)

            elif self.variant == "gini":
                # Concentration ratio: 1 - (unique_steps / total_tokens)
                # High = few unique steps for many tokens = synchronized = certain
                # Low = many unique steps = each token at its own time = spread
                concentration = 1.0 - (n_unique / n_tokens)
                result.append(concentration)

            else:
                raise ValueError(f"Unknown variant: {self.variant}")

        result = np.array(result)

        if self.variant == "entropy":
            # Direct: high entropy = spread commits = uncertain
            return result
        elif self.variant == "peak_ratio":
            # Negate: high peak ratio = synchronized = certain
            return -result
        elif self.variant == "gini":
            # Negate: high gini = concentrated = certain
            return -result
        return result


def load_estimator(cfg: Dict) -> Estimator:
    variant = cfg.get("variant", "entropy")
    return CommitmentSynchrony(variant=variant)
