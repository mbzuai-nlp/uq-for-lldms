from .semantic_matrix_trace import SemanticMatrixTraceCalculator
from lm_polygraph.defaults.stat_calculator_builders.utils import load_nli_model


def load_stat_calculator(config, builder):
    if not hasattr(builder, "nli_model"):
        builder.nli_model = load_nli_model(**config.nli_model)

    return SemanticMatrixTraceCalculator(builder.nli_model)
