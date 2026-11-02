import numpy as np
import torch
from typing import Dict
from lm_polygraph.estimators.estimator import Estimator


class MCMCDiagnostics(Estimator):
    """
    Applies MCMC convergence diagnostics to the per-token prediction
    trajectory across diffusion steps.

    Treats the sequence of log-probs for each token as a Markov chain
    and checks whether it has converged.

    Variants:
    - "split_rhat": Split-half R-hat (Gelman-Rubin). Compares variance
      of first half vs second half of the trajectory. R-hat ≈ 1 means
      converged; R-hat >> 1 means still changing. Detects non-stationarity.
    - "ess": Effective sample size via autocorrelation. Low ESS means
      the chain is highly autocorrelated — predictions are not independent,
      the model is "stuck". Normalized by chain length.
    """

    def __init__(self, variant: str = "split_rhat", aggregation: str = "avg"):
        super().__init__(["gen_trace_logits"], "sequence")
        self.variant = variant
        self.aggregation = aggregation

    def __str__(self):
        return f"MCMCDiag_{self.variant}_{self.aggregation}"

    def _compute_split_rhat(self, chain: torch.Tensor) -> float:
        """Split-half R-hat for a single chain."""
        n = len(chain)
        if n < 4:
            return 1.0  # Not enough data

        half = n // 2
        first = chain[:half].float()
        second = chain[half:2 * half].float()

        # Within-chain variance (average of two halves)
        w = (first.var() + second.var()) / 2

        # Between-chain variance
        grand_mean = (first.mean() + second.mean()) / 2
        b = half * ((first.mean() - grand_mean) ** 2 + (second.mean() - grand_mean) ** 2)

        # Posterior variance estimate
        var_hat = (1 - 1 / half) * w + b / half

        if w < 1e-10:
            return 1.0  # Constant chain = converged

        rhat = (var_hat / w).sqrt().item()
        return rhat

    def _compute_ess(self, chain: torch.Tensor) -> float:
        """Effective sample size via autocorrelation."""
        n = len(chain)
        if n < 4:
            return 1.0

        chain = chain.float()
        mean = chain.mean()
        var = chain.var()

        if var < 1e-10:
            return 1.0  # Constant chain

        # Compute autocorrelation at increasing lags
        max_lag = min(n // 2, 20)  # Cap at 20 lags for efficiency
        autocorr_sum = 0.0
        for lag in range(1, max_lag):
            c = ((chain[:n - lag] - mean) * (chain[lag:] - mean)).mean()
            rho = (c / var).item()
            if rho < 0.05:  # Stop when autocorrelation drops below threshold
                break
            autocorr_sum += rho

        ess = n / (1 + 2 * autocorr_sum)
        # Normalize by chain length: ESS/n in [0, 1], higher = more independent
        return min(ess / n, 1.0)

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        trace_logits = stats["gen_trace_logits"]  # (B, blocks, steps, block_len)

        B = trace_logits.shape[0]
        result = []

        for i in range(B):
            # (num_tokens, max_steps)
            logits_t = trace_logits[i].permute(0, 2, 1).reshape(-1, trace_logits.shape[2])

            token_scores = []
            for t in range(logits_t.shape[0]):
                chain = logits_t[t]
                valid = chain[torch.isfinite(chain)]

                if len(valid) < 4:
                    if self.variant == "split_rhat":
                        token_scores.append(1.0)
                    else:
                        token_scores.append(1.0)
                    continue

                if self.variant == "split_rhat":
                    token_scores.append(self._compute_split_rhat(valid))
                elif self.variant == "ess":
                    token_scores.append(self._compute_ess(valid))
                else:
                    raise ValueError(f"Unknown variant: {self.variant}")

            scores = np.array(token_scores)

            if self.aggregation == "avg":
                result.append(scores.mean())
            elif self.aggregation == "max":
                result.append(scores.max())
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")

        result = np.array(result)

        if self.variant == "split_rhat":
            # Direct: R-hat > 1 = not converged = uncertain
            return result
        elif self.variant == "ess":
            # Negate: high ESS = independent samples = converged = certain
            return -result
        return result


def load_estimator(cfg: Dict) -> Estimator:
    variant = cfg.get("variant", "split_rhat")
    aggregation = cfg.get("aggregation", "avg")
    return MCMCDiagnostics(variant=variant, aggregation=aggregation)
