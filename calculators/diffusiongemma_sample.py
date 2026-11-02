import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch

from lm_polygraph.stat_calculators.stat_calculator import StatCalculator


class DiffusionGemmaSamplingGenerationCalculator(StatCalculator):
    """Sampling-generation calculator for DiffusionGemma.

    Unlocks the sampling-based estimators (SemanticEntropy, KLE, LUQ, CoCoA-MSP) on the
    DiffusionGemma backend. It replaces lm-polygraph's :class:`SamplingGenerationCalculator`,
    which cannot run here because it does ``torch.stack(out.scores)`` while DiffusionGemma's
    ``generate()`` returns ``scores=None`` (the model has its own block-diffusion sampler and
    never exposes per-step AR scores), and because it reads ``out.hidden_states``, which the
    backend does not emit either.

    Instead we reuse what the backend already synthesizes: ``greedy_log_likelihoods`` is the
    per-position committed-token log-probability of the decode, i.e. the diffusion-native
    analogue of the AR per-token log-prob. Calling ``generate()`` K times therefore yields K
    decodes each with its own token ids, text, and per-token log-likelihoods, which is exactly
    the (texts, log-probs) pair the sampling estimators consume.

    Variation across the K decodes comes from the model itself: DiffusionGemma initialises
    the canvas with random tokens and renoises non-accepted positions, so decodes differ
    wherever the model is genuinely uncertain. Measured without any temperature override:
    5/5 distinct continuations on an open-ended prompt, 3/5 on an obscure date, 1/5 on a
    near-certain fact. The last case is correct calibrated behaviour, not a broken sampler,
    so a diversity check run only on easy factual prompts will look falsely degenerate.

    ``sample_temperature`` is therefore optional and defaults to off. Setting it attaches
    ``GumbelNoiseLogitsProcessor`` to widen the distribution, which is useful for a
    diversity ablation but draws the samples from a different decoding distribution than
    the greedy output. Leaving it off keeps the two comparable. The calculator warns if the
    decodes collapse to a single text, which would make the semantic estimators degenerate.

    Emitted stats match the built-in calculator's names so the downstream calculators
    (semantic matrices, greedy-vs-sample similarity) and the estimators run unchanged.
    Embeddings are not emitted: they are only needed by estimators outside this set, and the
    26B model makes them expensive to keep.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return [
            "sample_log_probs",
            "sample_log_likelihoods",
            "sample_tokens",
            "sample_texts",
        ], []

    def __init__(
        self,
        samples_n: int = 10,
        sample_temperature: float = None,
        warn_on_degenerate: bool = True,
    ):
        super().__init__()
        self.samples_n = int(samples_n)
        self.sample_temperature = (
            None if sample_temperature is None else float(sample_temperature)
        )
        self.warn_on_degenerate = bool(warn_on_degenerate)
        # Running collapse statistics, so the warning reflects the corpus, not one example.
        self._seen = 0
        self._collapsed = 0
        self._warned = False
        self._min_inputs_to_judge = 25
        self._collapse_warn_rate = 0.95

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}

        n_inputs = len(texts)
        sample_texts: List[List[str]] = [[] for _ in range(n_inputs)]
        sample_tokens: List[List[List[int]]] = [[] for _ in range(n_inputs)]
        sample_lls: List[List[List[float]]] = [[] for _ in range(n_inputs)]
        sample_log_probs: List[List[float]] = [[] for _ in range(n_inputs)]

        with torch.no_grad():
            for _ in range(self.samples_n):
                out = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=1,
                    num_return_sequences=1,
                    sample_temperature=self.sample_temperature,
                )
                # Never touch out.scores: DiffusionGemma returns None there.
                for i in range(n_inputs):
                    toks = list(out.greedy_tokens[i])
                    text = out.greedy_texts[i]
                    ll = [float(x) for x in out.greedy_log_likelihoods[i]]
                    if not toks:
                        warnings.warn(
                            f"DiffusionGemma produced an empty decode for input {i}; "
                            "skipping this sample."
                        )
                        continue
                    sample_tokens[i].append(toks)
                    sample_texts[i].append(text)
                    sample_lls[i].append(ll)
                    sample_log_probs[i].append(float(np.sum(ll)) if ll else 0.0)

        if self.warn_on_degenerate:
            self._seen += n_inputs
            self._collapsed += sum(
                1 for i in range(n_inputs)
                if len(sample_texts[i]) > 1 and len(set(sample_texts[i])) == 1
            )
            # A single input whose K decodes agree is expected: the model is simply
            # confident there, and collapsing on easy prompts is correct calibrated
            # behaviour rather than a broken sampler. Only a corpus-wide collapse means the
            # semantic estimators have nothing to measure, so warn on the running rate and
            # only once enough inputs have been seen to make the rate meaningful. Note runs
            # use batch_size=1, so a per-batch check would fire on every confident example.
            if self._seen >= self._min_inputs_to_judge and not self._warned:
                rate = self._collapsed / self._seen
                if rate >= self._collapse_warn_rate:
                    self._warned = True
                    warnings.warn(
                        f"{self._collapsed}/{self._seen} inputs returned K identical decodes "
                        f"({rate:.0%}). If this stays near 100%, SemanticEntropy and the "
                        "graph-based estimators have no variation to measure on this run.",
                        RuntimeWarning,
                    )

        return {
            "sample_texts": sample_texts,
            "sample_tokens": sample_tokens,
            "sample_log_likelihoods": sample_lls,
            "sample_log_probs": sample_log_probs,
        }
