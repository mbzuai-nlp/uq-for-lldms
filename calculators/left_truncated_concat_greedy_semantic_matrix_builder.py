from copy import copy
import logging

from lm_polygraph.defaults.stat_calculator_builders.utils import load_nli_model
from lm_polygraph.stat_calculators.greedy_semantic_matrix import (
    ConcatGreedySemanticMatrixCalculator,
)


log = logging.getLogger("lm_polygraph")


class _LeftTruncatingTokenizer:
    """Inject left-side truncation without changing the shared NLI tokenizer."""

    def __init__(self, tokenizer, max_length: int):
        self._tokenizer = copy(tokenizer)
        self._tokenizer.truncation_side = "left"
        self._max_length = max_length

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    def __call__(self, *args, **kwargs):
        kwargs["truncation"] = True
        kwargs["max_length"] = self._max_length
        return self._tokenizer(*args, **kwargs)


def load_stat_calculator(config, builder):
    if not hasattr(builder, "nli_model"):
        builder.nli_model = load_nli_model(**config.nli_model)

    max_length = int(getattr(config, "max_length", 1024))
    if max_length <= 0:
        raise ValueError("Concat NLI max_length must be positive")

    # Share model weights, but keep the tokenizer behavior local to this calculator.
    nli_model = copy(builder.nli_model)
    nli_model._deberta_tokenizer = _LeftTruncatingTokenizer(
        builder.nli_model.deberta_tokenizer,
        max_length=max_length,
    )
    log.info(
        "ConcatGreedySemanticMatrixCalculator uses left truncation at %d tokens",
        max_length,
    )
    return ConcatGreedySemanticMatrixCalculator(nli_model)
