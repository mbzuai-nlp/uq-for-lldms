from .greedy_deberta_similarity_trace import GreedyDebertaSimilarityMatrixTraceCalculator
from lm_polygraph.defaults.stat_calculator_builders.utils import load_nli_model


def load_stat_calculator(config, builder):
    """
    Builder entrypoint for GreedyDebertaSimilarityMatrixTraceCalculator.

    This creates a calculator that computes DeBERTa semantic similarity for the default
    generation trace. The NLI model is shared across all Deberta calculators
    through the builder environment to avoid reloading the model.

    Args:
        config: Configuration dictionary containing nli_model settings
        builder: Builder environment that may contain cached nli_model

    Returns:
        GreedyDebertaSimilarityMatrixTraceCalculator instance
    """
    if not hasattr(builder, "nli_model"):
        builder.nli_model = load_nli_model(**config.nli_model)

    return GreedyDebertaSimilarityMatrixTraceCalculator(builder.nli_model)
