from .greedy_cross_encoder_similarity_last_block_trace import GreedyCrossEncoderSimilarityMatrixLastBlockTraceCalculator


def load_stat_calculator(config, builder):
    # Get batch_size from config or use default
    batch_size = getattr(config, "batch_size", 10)
    cross_encoder_name = getattr(config, "cross_encoder_name", "cross-encoder/stsb-roberta-large")

    return GreedyCrossEncoderSimilarityMatrixLastBlockTraceCalculator(
        batch_size=batch_size,
        cross_encoder_name=cross_encoder_name,
    )
