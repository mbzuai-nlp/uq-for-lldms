from .greedy_deberta_similarity_last_block_trace import (
    GreedyDebertaSimilarityMatrixLastBlockTraceCalculator,
)
from lm_polygraph.defaults.stat_calculator_builders.utils import load_nli_model


def load_stat_calculator(config, builder):
    """
    Builder entrypoint for GreedyDebertaSimilarityMatrixLastBlockTraceCalculator.

    This creates a calculator that computes DeBERTa semantic similarity for the last block
    of the generation trace. The NLI model is shared across all Deberta calculators
    through the builder environment to avoid reloading the model.

    Args:
        config: Configuration dictionary containing nli_model settings
        builder: Builder environment that may contain cached nli_model

    Returns:
        GreedyDebertaSimilarityMatrixLastBlockTraceCalculator instance
    """
    if not hasattr(builder, "nli_model"):
        builder.nli_model = load_nli_model(**config.nli_model)

    return GreedyDebertaSimilarityMatrixLastBlockTraceCalculator(builder.nli_model)
