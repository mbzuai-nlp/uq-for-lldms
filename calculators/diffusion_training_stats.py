import gc
import logging
import torch
import numpy as np

from typing import Dict, List, Tuple
from lm_polygraph.generation_metrics.generation_metric import GenerationMetric
from lm_polygraph.stat_calculators.attention import (
    AttentionFeaturesCalculator,
    LookbackRatioCalculator,
)
from lm_polygraph.stat_calculators.embeddings import (
    EmbeddingsCalculator,
    TokenEmbeddingsCalculator,
)
from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel

from .diffusion_greedy_probs import DiffusionGreedyProbsCalculator


log = logging.getLogger("lm_polygraph")


class DiffusionTrainingStatisticExtractionCalculator(StatCalculator):
    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return [
            "train_greedy_texts",
            "train_input_tokens",
            "train_embeddings",
            "train_token_embeddings",
            "background_train_embeddings",
            "background_train_token_embeddings",
            "train_greedy_tokens",
            "train_greedy_log_likelihoods",
            "train_input_embeddings_mean",
            "train_gen_trace_embeddings",
            "train_gen_trace_logits",
            "train_gen_trace_entropy",
            "train_gen_trace_argmax",
            "train_gen_trace_top2gap",
            "train_generation_trace",
            "train_full_generation_trace",
            "train_lookback_ratios",
            "train_attention_features",
            "train_metrics",
        ], []

    def __init__(
        self,
        train_dataset=None,
        background_train_dataset=None,
        output_attentions: bool = True,
        output_hidden_states: bool = True,
        output_diffusion_stats: bool = True,
        output_token_embeddings: bool = False,
        hidden_state_layer=None,
        return_embeddings: bool = False,
        return_token_embeddings: bool = False,
        return_lookback_ratios: bool = False,
        return_attention_features: bool = False,
        target_metric: GenerationMetric = None,
    ):
        # Adapted from lm-polygraph supervised_methods statistic_extraction.py: one-shot train statistics extraction.
        super().__init__()
        self.train_dataset = train_dataset
        self.background_train_dataset = background_train_dataset
        self.statistics_extracted = False
        self.base_calculators = [
            DiffusionGreedyProbsCalculator(
                output_attentions=(
                    output_attentions
                    or return_lookback_ratios
                    or return_attention_features
                ),
                output_hidden_states=(
                    output_hidden_states
                    or return_embeddings
                    or return_token_embeddings
                ),
                output_diffusion_stats=output_diffusion_stats,
                output_token_embeddings=output_token_embeddings,
                hidden_state_layer=hidden_state_layer,
            )
        ]
        if return_embeddings:
            self.base_calculators.append(EmbeddingsCalculator())
        if return_token_embeddings:
            self.base_calculators.append(TokenEmbeddingsCalculator())
        if return_lookback_ratios:
            self.base_calculators.append(LookbackRatioCalculator())
        if return_attention_features:
            self.base_calculators.append(AttentionFeaturesCalculator())
        if target_metric is not None:
            self.target_metric = target_metric

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
        background_train_dataset_max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        # Adapted from lm-polygraph supervised_methods statistic_extraction.py: iterates train/background datasets once.
        if self.statistics_extracted:
            return {}
        else:
            train_stats = {}
            result_train_stat = {}
            datasets = [self.train_dataset, self.background_train_dataset]
            datasets_name = ["train_", "background_train_"]
            skip_keywords = ["tokenizer", "layers", "_raw"]
            for dataset, dataset_name in zip(datasets, datasets_name):
                if dataset is None:
                    continue
                train_max_new_tokens = (
                    max_new_tokens
                    if dataset_name == "train_"
                    else background_train_dataset_max_new_tokens
                )
                for batch_i, batch in enumerate(dataset):
                    if len(batch) == 3:
                        inp_texts, target_texts, images = batch
                    elif len(batch) == 2:
                        inp_texts, target_texts = batch
                    else:
                        raise ValueError(
                            f"Expected batch with 2 or 3 elements, got {len(batch)}"
                        )
                    batch_stats: Dict[str, np.ndarray] = {}
                    for key, val in [
                        ("input_texts", inp_texts),
                        ("target_texts", target_texts),
                    ]:
                        batch_stats[key] = val
                        batch_stats["layers"] = dependencies.get("layers", [-1])

                    log.info(
                        "Diffusion train stats: %sbatch %s started",
                        dataset_name,
                        batch_i,
                    )
                    for stat_calculator in self.base_calculators:
                        calc_name = stat_calculator.__class__.__name__
                        log.info(
                            "Diffusion train stats: %sbatch %s %s started",
                            dataset_name,
                            batch_i,
                            calc_name,
                        )
                        try:
                            new_stats = stat_calculator(
                                batch_stats, inp_texts, model, train_max_new_tokens
                            )
                        except ZeroDivisionError:
                            if not isinstance(stat_calculator, LookbackRatioCalculator):
                                raise
                            new_stats = self._safe_lookback_ratios(batch_stats, model)
                            log.info(
                                "Diffusion train stats: %sbatch %s %s used zero-denominator fallback",
                                dataset_name,
                                batch_i,
                                calc_name,
                            )
                        for stat, stat_value in new_stats.items():
                            if stat in batch_stats.keys():
                                continue
                            batch_stats[stat] = stat_value
                        log.info(
                            "Diffusion train stats: %sbatch %s %s finished",
                            dataset_name,
                            batch_i,
                            calc_name,
                        )

                    if dataset_name == "train_":
                        log.info(
                            "Diffusion train stats: %sbatch %s target metric started",
                            dataset_name,
                            batch_i,
                        )
                        batch_stats["metrics"] = self.target_metric(
                            batch_stats, target_texts
                        )
                        log.info(
                            "Diffusion train stats: %sbatch %s target metric finished",
                            dataset_name,
                            batch_i,
                        )

                    for stat in batch_stats.keys():
                        if stat in [
                            "input_tokens",
                            "input_texts",
                            "target_texts",
                        ] or any(keyword in stat for keyword in skip_keywords):
                            continue
                        if dataset_name + stat in train_stats.keys():
                            train_stats[dataset_name + stat].append(batch_stats[stat])
                        else:
                            train_stats[dataset_name + stat] = [batch_stats[stat]]

                    torch.cuda.empty_cache()
                    gc.collect()

            for stat in train_stats.keys():
                if any(s is None for s in train_stats[stat]) or any(
                    keyword in stat for keyword in skip_keywords
                ):
                    continue
                result_train_stat[stat] = self._finalize_stat(
                    stat, train_stats[stat]
                )
            self.statistics_extracted = True

            return result_train_stat

    def _safe_lookback_ratios(self, dependencies, model):
        # Adapted from lm-polygraph stat_calculators/attention.py: preserves lookback ratios when attention sums are zero.
        attentions = dependencies["attention_raw"]
        cut_sequences = dependencies["greedy_tokens"]

        lookback_ratios = []
        if model.model_type != "vLLMCausalLM":
            for i in range(len(cut_sequences)):
                c = len(cut_sequences[i])
                for j in range(c):
                    lookback_ratios_token = []
                    for layer in range(len(attentions[j])):
                        for head in range(len(attentions[j][layer][0])):
                            if j:
                                attention_on_new = (
                                    attentions[j][layer][0][head][0][-j:].mean().item()
                                )
                                attention_on_context = (
                                    attentions[j][layer][0][head][0][:-j].mean().item()
                                )
                            else:
                                attention_on_new = 0
                                attention_on_context = (
                                    attentions[j][layer][0][head][0].mean().item()
                                )
                            denom = attention_on_new + attention_on_context
                            if not np.isfinite(denom) or denom == 0:
                                lookback_ratio = 0.0
                            else:
                                lookback_ratio = attention_on_context / denom
                            lookback_ratios_token.append(lookback_ratio)
                    lookback_ratios.append(lookback_ratios_token)
        return {"lookback_ratios": lookback_ratios}

    def _finalize_stat(self, stat, values):
        # Adapted from lm-polygraph supervised_methods statistic_extraction.py: concatenates collected train stats.
        first = values[0]
        if isinstance(first, torch.Tensor):
            return self._concat_tensors(stat, values)
        if isinstance(first, np.ndarray):
            return np.concatenate(values)
        if isinstance(first, list):
            return [item for value in values for item in value]
        return np.array(values)

    def _concat_tensors(self, stat, values):
        # Adapted from TraceDet DLLM_generate/generate_interoutput.py: pads trace tensors with sentinel values before saving.
        if all(value.shape[1:] == values[0].shape[1:] for value in values):
            return torch.cat(values, dim=0)

        max_shape = tuple(
            max(value.shape[dim] for value in values)
            for dim in range(1, values[0].ndim)
        )
        pad_value = -1 if stat.endswith("argmax") else float("-inf")
        if not torch.is_floating_point(values[0]) and pad_value == float("-inf"):
            pad_value = 0

        padded = []
        for value in values:
            target = value.new_full((value.shape[0], *max_shape), pad_value)
            slices = (slice(None),) + tuple(
                slice(0, size) for size in value.shape[1:]
            )
            target[slices] = value
            padded.append(target)
        return torch.cat(padded, dim=0)
