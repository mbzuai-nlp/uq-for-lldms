from .greedy_deberta_similarity_full_trace import GreedyDebertaSimilarityMatrixFullTraceCalculator
from lm_polygraph.defaults.stat_calculator_builders.utils import load_nli_model


def load_stat_calculator(config, builder):
    if not hasattr(builder, "nli_model"):
        builder.nli_model = load_nli_model(**config.nli_model)

    return GreedyDebertaSimilarityMatrixFullTraceCalculator(builder.nli_model)
