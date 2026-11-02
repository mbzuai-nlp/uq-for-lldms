import numpy as np
import torch

from typing import List, Dict, Optional

from lm_polygraph.estimators.estimator import Estimator


class CocoaMCMaskedNLLTrace(Estimator):
    def __init__(
        self,
        inf_method: str = "MCMaskedNLL",
        progress_score: bool = False,
        multiply_by_nfe: bool = False,
        full_trace: bool = False,
        last_block_only: bool = False,
        block_only: bool = False,
        block_agg: str = "avg",
        inverse_progress_score: bool = False,
        similarity_metric: str = "cross_encoder",
    ):
        self.inf_method = inf_method
        self.progress_score = progress_score
        self.multiply_by_nfe = multiply_by_nfe
        self.full_trace = full_trace
        self.last_block_only = last_block_only
        self.block_only = block_only
        self.block_agg = block_agg
        self.inverse_progress_score = inverse_progress_score
        self.similarity_metric = similarity_metric

        if self.inf_method == "MCMaskedNLL":
            self.inf_stat = "mc_masked_ll"
        elif self.inf_method == "MCMaskedNormalizedNLL":
            self.inf_stat = "mc_masked_normalized_ll"
        elif self.inf_method == "NFE":
            self.inf_stat = "nfe"
        elif self.inf_method == "TraceProbs":
            self.inf_stat = "gen_trace_logits"
        elif self.inf_method == "TraceEntropy":
            self.inf_stat = "gen_trace_entropy"
        elif self.inf_method == "BlockMCMaskedNLL":
            self.inf_stat = "block_mc_masked_ll"
        elif self.inf_method == "ConfidenceAtCommit":
            self.inf_stat = "confidence_at_commit"

        # Determine which stats to use based on similarity_metric and trace flags
        if similarity_metric == "deberta_entail":
            if block_only:
                similarity_trace = "greedy_deberta_entail_block"
            elif last_block_only:
                similarity_trace = "greedy_deberta_entail_last_block_trace"
            elif full_trace:
                similarity_trace = "greedy_deberta_entail_full_trace"
            else:
                similarity_trace = "greedy_deberta_entail_trace"
        elif similarity_metric == "deberta_contra":
            if block_only:
                similarity_trace = "greedy_deberta_contra_block"
            elif last_block_only:
                similarity_trace = "greedy_deberta_contra_last_block_trace"
            elif full_trace:
                similarity_trace = "greedy_deberta_contra_full_trace"
            else:
                similarity_trace = "greedy_deberta_contra_trace"
        else:  # cross_encoder (default)
            if block_only:
                similarity_trace = "greedy_sentence_similarity_block"
            elif last_block_only:
                similarity_trace = "greedy_sentence_similarity_last_block_trace"
            elif full_trace:
                similarity_trace = "greedy_sentence_similarity_full_trace"
            else:
                similarity_trace = "greedy_sentence_similarity_trace"

        stats_dependencies = [similarity_trace, self.inf_stat, "greedy_texts"]
        if self.multiply_by_nfe:
            stats_dependencies.append("nfe")

        super().__init__(stats_dependencies, "sequence")
        self.similarity_trace = similarity_trace

    def __str__(self):
        progress = "Progress" if self.progress_score else ""
        inverse_progress = "InverseProgress" if self.inverse_progress_score else ""
        nfe_str = "NFE" if self.multiply_by_nfe else ""
        full = "Full" if self.full_trace else ""
        last_block = "LastBlock" if self.last_block_only else ""
        block = "Block" if self.block_only else ""
        block_agg = self.block_agg.capitalize() if self.block_only else ""

        metric_suffix = ""
        if self.similarity_metric == "deberta_entail":
            metric_suffix = "DebertaEntail"
        elif self.similarity_metric == "deberta_contra":
            metric_suffix = "DebertaContra"
        else:  # cross_encoder (default)
            metric_suffix = "CrossEncoder"

        return f"Cocoa{self.inf_method}{metric_suffix}{full}{last_block}{block}{block_agg}Trace{progress}{inverse_progress}{nfe_str}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        greedy_texts = stats["greedy_texts"]  # Reference for expected batch size
        expected_size = len(greedy_texts)

        batch_greedy_sentence_similarity = stats[self.similarity_trace]

        # Special handling for TraceProbs and TraceEntropy inf_method
        if self.inf_method in ["TraceProbs", "TraceEntropy"]:
            # Process gen_trace_logits or gen_trace_entropy using same logic as trace_probs/trace_entropy
            if self.inf_method == "TraceProbs":
                trace_tensor = stats["gen_trace_logits"]  # (B, num_blocks, max_steps, block_length)
                # Sum over tokens (last dimension), negate to get positive log-probs
                trace_flat = trace_tensor.sum(dim=-1)  # (B, num_blocks, max_steps)
            else:  # TraceEntropy
                trace_tensor = stats["gen_trace_entropy"]  # (B, num_blocks, max_steps, block_length)
                # Mean over tokens (last dimension)
                trace_flat = -trace_tensor.mean(dim=-1)  # (B, num_blocks, max_steps)

            trace_flat = trace_flat.reshape(trace_flat.shape[0], -1)  # (B, num_blocks * max_steps)

            # Compute batch_lls from trace tensor
            batch_lls = []
            for i in range(trace_flat.shape[0]):
                valid_mask = torch.isfinite(trace_flat[i])
                if valid_mask.any():
                    valid_values = trace_flat[i][valid_mask]
                    # Use avg aggregation for cocoa
                    batch_lls.append(valid_values.mean().item())
                else:
                    batch_lls.append(0.0)
            batch_lls = np.array(batch_lls)
        else:
            # Special handling for ConfidenceAtCommit
            if self.inf_method == "ConfidenceAtCommit":
                conf = stats["confidence_at_commit"]  # (B, gen_length)
                batch_lls = []
                for i in range(conf.shape[0]):
                    valid_mask = torch.isfinite(conf[i])
                    valid = conf[i][valid_mask]
                    if len(valid) > 0:
                        batch_lls.append(valid.mean().item())
                    else:
                        batch_lls.append(0.0)
                batch_lls = np.array(batch_lls)
            else:
                batch_lls = np.array(stats[self.inf_stat])

        # For contradiction, higher values mean more dissimilarity, so we don't invert
        use_inverse = not ("contra" in self.similarity_trace)

        enriched_metrics = []  # To store enriched metrics for each sample

        # Handle per-block structure: [bs][block][step]
        if self.block_only:
            for idx, (greedy_ll, blocks_similarity) in enumerate(
                zip(batch_lls, batch_greedy_sentence_similarity)
            ):
                if not isinstance(blocks_similarity, list) or len(blocks_similarity) == 0:
                    # For empty similarity, use neutral uncertainty value
                    enriched_metric = (-greedy_ll if self.inf_method != "NFE" else greedy_ll)
                    if self.multiply_by_nfe:
                        enriched_metric *= stats["nfe"][idx]
                    enriched_metrics.append(enriched_metric)
                    continue

                # Compute probabilities (negative log-probs)
                if self.inf_method == "NFE":
                    prob = greedy_ll  # For NFE, the "probability" is just the NFE value
                else:
                    prob = -greedy_ll

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
                    avg_dissimilarity = 1.0  # Maximum dissimilarity

                enriched_metric = prob * avg_dissimilarity
                if self.multiply_by_nfe:
                    enriched_metric *= stats["nfe"][idx]
                enriched_metrics.append(enriched_metric)
        else:
            # Original logic for non-block structure
            for idx, (greedy_ll, greedy_sentence_similarity) in enumerate(
                zip(batch_lls, batch_greedy_sentence_similarity)
            ):
                # Handle empty similarity arrays
                if len(greedy_sentence_similarity) == 0:
                    # For empty similarity, use neutral uncertainty value
                    enriched_metric = (-greedy_ll if self.inf_method != "NFE" else greedy_ll)
                    if self.multiply_by_nfe:
                        enriched_metric *= stats["nfe"][idx]
                    enriched_metrics.append(enriched_metric)
                    continue

                # Compute probabilities (negative log-probs)
                if self.inf_method == "NFE":
                    prob = greedy_ll  # For NFE, the "probability" is just the NFE value
                else:
                    prob = -greedy_ll

                # Compute row-wise average similarity, excluding self-similarity
                # Diagonal contains self-similarities
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

                enriched_metric = prob * avg_dissimilarity
                if self.multiply_by_nfe:
                    enriched_metric *= stats["nfe"][idx]
                enriched_metrics.append(enriched_metric)

        result = np.array(enriched_metrics)

        # Pad with zeros if we got fewer results than expected
        if len(result) < expected_size:
            padding = np.zeros(expected_size - len(result), dtype=result.dtype)
            result = np.concatenate([result, padding])

        return result


def load_estimator(cfg: Dict) -> Estimator:
    """
    Factory function to create a `CocoaMCMaskedNLLTrace` estimator from a config dictionary.

    Args:
        cfg: Configuration dictionary with the following optional keys:
            - inf_method (str): Inference method. Options: "MCMaskedNLL", "MCMaskedNormalizedNLL", "NFE", "TraceProbs", "TraceEntropy", "BlockMCMaskedNLL"
            - progress_score (bool): Whether to use progress-weighted similarity scores (later steps have higher weight)
            - multiply_by_nfe (bool): Whether to multiply the result by NFE
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
    inf_method = cfg.get("inf_method", "MCMaskedNLL")
    progress_score = cfg.get("progress_score", False)
    multiply_by_nfe = cfg.get("multiply_by_nfe", False)
    full_trace = cfg.get("full_trace", False)
    last_block_only = cfg.get("last_block_only", False)
    block_only = cfg.get("block_only", False)
    block_agg = cfg.get("block_agg", "avg")
    inverse_progress_score = cfg.get("inverse_progress_score", False)
    similarity_metric = cfg.get("similarity_metric", "cross_encoder")

    return CocoaMCMaskedNLLTrace(
        inf_method=inf_method,
        progress_score=progress_score,
        multiply_by_nfe=multiply_by_nfe,
        full_trace=full_trace,
        last_block_only=last_block_only,
        block_only=block_only,
        block_agg=block_agg,
        inverse_progress_score=inverse_progress_score,
        similarity_metric=similarity_metric,
    )
