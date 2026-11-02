import numpy as np

from typing import List, Dict, Optional

from lm_polygraph.estimators.estimator import Estimator


class AvgDissimilarityTrace(Estimator):
    def __init__(
        self,
        progress_score: bool = False,
        full_trace: bool = False,
        last_block_only: bool = False,
        block_only: bool = False,
        block_agg: str = "avg",
        inverse_progress_score: bool = False,
        similarity_metric: str = "cross_encoder"
    ):
        """
        Args:
            progress_score: Whether to use progress-weighted similarity scores
            full_trace: Whether to use full generation trace
            last_block_only: Whether to use last block generation trace only
            block_only: Whether to use per-block generation traces
            block_agg: How to aggregate block dissimilarities: "avg" or "max"
            inverse_progress_score: Whether to use inverse progress weights (earlier steps have higher weight)
            similarity_metric: Type of similarity metric to use. Options:
                - "cross_encoder": Cross-encoder similarity scores (default)
                - "deberta_entail": DeBERTa entailment probability
                - "deberta_contra": DeBERTa contradiction probability
        """
        self.similarity_metric = similarity_metric
        self.progress_score = progress_score
        self.full_trace = full_trace
        self.last_block_only = last_block_only
        self.block_only = block_only
        self.block_agg = block_agg
        self.inverse_progress_score = inverse_progress_score

        # Determine which stats to use based on similarity_metric
        if similarity_metric == "deberta_entail":
            if block_only:
                stat_name = "greedy_deberta_entail_block"
            elif last_block_only:
                stat_name = "greedy_deberta_entail_last_block_trace"
            elif full_trace:
                stat_name = "greedy_deberta_entail_full_trace"
            else:
                stat_name = "greedy_deberta_entail_trace"
        elif similarity_metric == "deberta_contra":
            if block_only:
                stat_name = "greedy_deberta_contra_block"
            elif last_block_only:
                stat_name = "greedy_deberta_contra_last_block_trace"
            elif full_trace:
                stat_name = "greedy_deberta_contra_full_trace"
            else:
                stat_name = "greedy_deberta_contra_trace"
        else:  # cross_encoder (default)
            if block_only:
                stat_name = "greedy_sentence_similarity_block"
            elif last_block_only:
                stat_name = "greedy_sentence_similarity_last_block_trace"
            elif full_trace:
                stat_name = "greedy_sentence_similarity_full_trace"
            else:
                stat_name = "greedy_sentence_similarity_trace"

        super().__init__(
            [
                stat_name,
                "greedy_texts",  # Add to get expected batch size
            ],
            "sequence",
        )
        self.stat_name = stat_name

    def __str__(self):
        progress = "Progress" if self.progress_score else ""
        inverse_progress = "InverseProgress" if self.inverse_progress_score else ""
        full = "Full" if self.full_trace else ""
        last_block = "LastBlock" if self.last_block_only else ""
        block = "Block" if self.block_only else ""
        block_agg = self.block_agg.capitalize() if self.block_only else ""

        metric_suffix = ""
        if self.similarity_metric == "deberta_entail":
            metric_suffix = "DebertaEntail"
        elif self.similarity_metric == "deberta_contra":
            metric_suffix = "DebertaContra"

        return f"AvgDissimilarity{metric_suffix}{full}{last_block}{block}{block_agg}Trace{progress}{inverse_progress}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        greedy_texts = stats["greedy_texts"]  # Reference for expected batch size
        expected_size = len(greedy_texts)

        batch_greedy_sentence_similarity = stats[self.stat_name]

        # For contradiction, higher values mean more dissimilarity, so we don't invert
        use_inverse = not ("contra" in self.stat_name)

        enriched_metrics = []  # To store enriched metrics for each sample

        # Handle per-block structure: [bs][block][step]
        if self.block_only:
            for blocks_similarity in batch_greedy_sentence_similarity:
                if len(blocks_similarity) == 0:
                    enriched_metrics.append(0.0)
                    continue

                # For each block, compute average dissimilarity
                block_avg_dissims = []
                for block_similarity in blocks_similarity:
                    if len(block_similarity) == 0:
                        continue

                    if self.progress_score or self.inverse_progress_score:
                        if self.inverse_progress_score:
                            # Inverse progress: earlier steps have higher weight
                            progress_weights = (np.arange(len(block_similarity)) + 1) / len(block_similarity)
                        else:
                            # Normal progress: later steps have higher weight
                            progress_weights = (np.arange(len(block_similarity)) + 1)[::-1] / len(block_similarity)
                        progress_scores = block_similarity * progress_weights
                        if use_inverse:
                            avg_dissim = 1.0 - np.mean(progress_scores)
                        else:
                            avg_dissim = np.mean(progress_scores)
                    else:
                        if use_inverse:
                            avg_dissim = np.mean(1 - block_similarity)
                        else:
                            avg_dissim = np.mean(block_similarity)
                    block_avg_dissims.append(avg_dissim)

                # Aggregate block averages
                if block_avg_dissims:
                    if self.block_agg == "avg":
                        avg_dissimilarity = np.mean(block_avg_dissims)
                    else:  # "max"
                        avg_dissimilarity = np.max(block_avg_dissims)
                else:
                    avg_dissimilarity = 0.0

                enriched_metrics.append(avg_dissimilarity)
        else:
            # Original logic for non-block structure
            for greedy_sentence_similarity in batch_greedy_sentence_similarity:
                if len(greedy_sentence_similarity) == 0:
                    enriched_metrics.append(0.0)  # Neutral value for empty similarity
                    continue

                if self.progress_score or self.inverse_progress_score:
                    if self.inverse_progress_score:
                        # Inverse progress: earlier steps have higher weight
                        progress_weights = (np.arange(len(greedy_sentence_similarity)) + 1) / len(greedy_sentence_similarity)
                    else:
                        # Normal progress: later steps have higher weight
                        progress_weights = (np.arange(len(greedy_sentence_similarity)) + 1)[::-1] / len(greedy_sentence_similarity)
                    progress_scores = greedy_sentence_similarity * progress_weights
                    if use_inverse:
                        avg_dissimilarity = 1.0 - np.mean(progress_scores)
                    else:
                        avg_dissimilarity = np.mean(progress_scores)
                else:
                    if use_inverse:
                        avg_dissimilarity = np.mean(1 - greedy_sentence_similarity)
                    else:
                        avg_dissimilarity = np.mean(greedy_sentence_similarity)
                enriched_metrics.append(avg_dissimilarity)

        result = np.array(enriched_metrics)

        # Pad with zeros if we got fewer results than expected
        if len(result) < expected_size:
            padding = np.zeros(expected_size - len(result), dtype=result.dtype)
            result = np.concatenate([result, padding])

        return result


def load_estimator(cfg: Dict) -> Estimator:
    """
    Factory function to create a `AvgDissimilarityTrace` estimator from a config dictionary.

    Args:
        cfg: Configuration dictionary with the following optional keys:
            - progress_score (bool): Whether to use progress-weighted similarity scores (later steps have higher weight)
            - full_trace (bool): Whether to use full generation trace
            - last_block_only (bool): Whether to use last block generation trace only
            - block_only (bool): Whether to use per-block generation traces
            - block_agg (str): How to aggregate block dissimilarities: "avg" or "max"
            - inverse_progress_score (bool): Whether to use inverse progress weights (earlier steps have higher weight)
            - similarity_metric (str): Type of similarity metric. Options:
                * "cross_encoder": Cross-encoder similarity scores (default)
                * "deberta_entail": DeBERTa entailment probability
                * "deberta_contra": DeBERTa contradiction probability
    """
    progress_score = cfg.get("progress_score", False)
    full_trace = cfg.get("full_trace", False)
    last_block_only = cfg.get("last_block_only", False)
    block_only = cfg.get("block_only", False)
    block_agg = cfg.get("block_agg", "avg")
    inverse_progress_score = cfg.get("inverse_progress_score", False)
    similarity_metric = cfg.get("similarity_metric", "cross_encoder")

    return AvgDissimilarityTrace(
        progress_score=progress_score,
        full_trace=full_trace,
        last_block_only=last_block_only,
        block_only=block_only,
        block_agg=block_agg,
        inverse_progress_score=inverse_progress_score,
        similarity_metric=similarity_metric
    )
