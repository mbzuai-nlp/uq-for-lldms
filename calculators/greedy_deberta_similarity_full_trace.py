import numpy as np

import itertools
from typing import Dict, List, Tuple
from tqdm import tqdm

from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
import torch.nn as nn
import torch

softmax = nn.Softmax(dim=1)


class GreedyDebertaSimilarityMatrixFullTraceCalculator(StatCalculator):
    """
    Calculates the DeBERTa semantic similarity between greedy sequence and sampled sequences.
    Uses entailment probability as the similarity metric.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return (
            [
                "greedy_deberta_entail_forward_full_trace",
                "greedy_deberta_entail_backward_full_trace",
                "greedy_deberta_entail_full_trace",
                "greedy_deberta_contra_forward_full_trace",
                "greedy_deberta_contra_backward_full_trace",
                "greedy_deberta_contra_full_trace",
                "greedy_deberta_classes_forward_full_trace",
                "greedy_deberta_classes_backward_full_trace",
                "greedy_deberta_classes_full_trace",
            ],
            ["input_texts", "full_generation_trace", "greedy_texts"],
        )

    def __init__(self, nli_model):
        super().__init__()
        self.is_deberta_setup = False
        self.nli_model = nli_model

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the DeBERTa semantic similarity between greedy sequence and sampled sequences.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, containing:
                - 'full_generation_trace' (List[List[str]]): several sampling generations
                    for each input text in the batch.
                - 'greedy_texts' (List[str]): greedy generation for each input text in the batch.
            texts (List[str]): Input texts batch used for model generation.
            model (WhiteboxModel): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'greedy_deberta_entail_forward_full_trace' (np.ndarray): entailment probabilities
                    for pairs (greedy_text, sample_text).
                - 'greedy_deberta_entail_backward_full_trace' (np.ndarray): entailment probabilities
                    for pairs (sample_text, greedy_text).
                - 'greedy_deberta_entail_full_trace' (np.ndarray): average entailment probabilities.
                - 'greedy_deberta_contra_forward_full_trace' (np.ndarray): contradiction probabilities
                    for pairs (greedy_text, sample_text).
                - 'greedy_deberta_contra_backward_full_trace' (np.ndarray): contradiction probabilities
                    for pairs (sample_text, greedy_text).
                - 'greedy_deberta_contra_full_trace' (np.ndarray): average contradiction probabilities.
                - 'greedy_deberta_classes_forward_full_trace' (np.ndarray): predicted class ids
                    for pairs (greedy_text, sample_text).
                - 'greedy_deberta_classes_backward_full_trace' (np.ndarray): predicted class ids
                    for pairs (sample_text, greedy_text).
                - 'greedy_deberta_classes_full_trace' (np.ndarray): average predicted class ids.
        """

        deberta = self.nli_model
        deberta_batch_size = deberta.batch_size
        batch_texts = dependencies["full_generation_trace"]
        batch_greedy_texts = dependencies["greedy_texts"]

        batch_pairs = []
        batch_pairs_b = []
        batch_invs = []
        for texts, greedy_text in zip(batch_texts, batch_greedy_texts):
            # Sampling from LLM often produces significant number of identical
            # outputs. We only need to score pairs of unqiue outputs
            unique_texts, inv = np.unique(texts, return_inverse=True)
            batch_pairs.append(list(itertools.product([greedy_text], unique_texts)))
            batch_pairs_b.append(list(itertools.product(unique_texts, [greedy_text])))
            batch_invs.append(inv)

        device = deberta.device
        ent_id = deberta.deberta.config.label2id["ENTAILMENT"]
        contra_id = deberta.deberta.config.label2id["CONTRADICTION"]

        tokenizer = deberta.deberta_tokenizer

        entail_arrays_f = []
        entail_arrays_b = []
        entail_arrays = []
        contra_arrays_f = []
        contra_arrays_b = []
        contra_arrays = []
        classes_arrays_f = []
        classes_arrays_b = []
        classes_arrays = []

        with torch.no_grad():
            for i, pairs in enumerate(batch_pairs):
                pairs_b = batch_pairs_b[i]
                inv = batch_invs[i]

                # Handle empty pairs - add zero similarity to maintain batch size
                if len(pairs) == 0:
                    # Add single zero similarity for this batch element
                    entail_arrays_f.append(np.array([0.0]))
                    entail_arrays_b.append(np.array([0.0]))
                    entail_arrays.append(np.array([0.0]))
                    contra_arrays_f.append(np.array([0.0]))
                    contra_arrays_b.append(np.array([0.0]))
                    contra_arrays.append(np.array([0.0]))
                    classes_arrays_f.append(np.array([0]))
                    classes_arrays_b.append(np.array([0]))
                    classes_arrays.append(np.array([0]))
                    continue

                # Process forward pairs (greedy_text, sample_text)
                dl_f = torch.utils.data.DataLoader(pairs, batch_size=deberta_batch_size)
                probs_f = []
                logits_f = []
                for first_texts, second_texts in dl_f:
                    encoded = tokenizer(
                        list(first_texts),
                        list(second_texts),
                        padding=True,
                        return_tensors="pt",
                    ).to(device)
                    logits = deberta.deberta(**encoded).logits
                    probs_f.append(softmax(logits))
                    logits_f.append(logits)

                probs_f = torch.cat(probs_f, dim=0)
                logits_f = torch.cat(logits_f, dim=0)

                del encoded, logits
                torch.cuda.empty_cache()

                entail_probs_f = probs_f[:, ent_id]
                contra_probs_f = probs_f[:, contra_id]
                class_preds_f = probs_f.argmax(-1)

                # Process backward pairs (sample_text, greedy_text)
                dl_b = torch.utils.data.DataLoader(pairs_b, batch_size=deberta_batch_size)
                probs_b = []
                logits_b = []
                for first_texts, second_texts in dl_b:
                    encoded = tokenizer(
                        list(first_texts),
                        list(second_texts),
                        padding=True,
                        return_tensors="pt",
                    ).to(device)
                    logits = deberta.deberta(**encoded).logits
                    probs_b.append(softmax(logits))
                    logits_b.append(logits)

                probs_b = torch.cat(probs_b, dim=0)
                logits_b = torch.cat(logits_b, dim=0)

                del encoded, logits
                torch.cuda.empty_cache()

                entail_probs_b = probs_b[:, ent_id]
                contra_probs_b = probs_b[:, contra_id]
                class_preds_b = probs_b.argmax(-1)

                # Convert to numpy and apply inverse indices
                entail_arrays_f.append(entail_probs_f.cpu().numpy()[inv])
                entail_arrays_b.append(entail_probs_b.cpu().numpy()[inv])
                entail_arrays.append(
                    (entail_probs_f.cpu().numpy()[inv] + entail_probs_b.cpu().numpy()[inv]) / 2
                )

                contra_arrays_f.append(contra_probs_f.cpu().numpy()[inv])
                contra_arrays_b.append(contra_probs_b.cpu().numpy()[inv])
                contra_arrays.append(
                    (contra_probs_f.cpu().numpy()[inv] + contra_probs_b.cpu().numpy()[inv]) / 2
                )

                classes_arrays_f.append(class_preds_f.cpu().numpy()[inv])
                classes_arrays_b.append(class_preds_b.cpu().numpy()[inv])
                classes_arrays.append(
                    (class_preds_f.cpu().numpy()[inv] + class_preds_b.cpu().numpy()[inv]) / 2
                )

        if not entail_arrays_f:
            return {
                "greedy_deberta_entail_forward_full_trace": np.array([]),
                "greedy_deberta_entail_backward_full_trace": np.array([]),
                "greedy_deberta_entail_full_trace": np.array([]),
                "greedy_deberta_contra_forward_full_trace": np.array([]),
                "greedy_deberta_contra_backward_full_trace": np.array([]),
                "greedy_deberta_contra_full_trace": np.array([]),
                "greedy_deberta_classes_forward_full_trace": np.array([]),
                "greedy_deberta_classes_backward_full_trace": np.array([]),
                "greedy_deberta_classes_full_trace": np.array([]),
            }

        entail_arrays_f = np.stack(entail_arrays_f)
        entail_arrays_b = np.stack(entail_arrays_b)
        entail_arrays = np.stack(entail_arrays)

        contra_arrays_f = np.stack(contra_arrays_f)
        contra_arrays_b = np.stack(contra_arrays_b)
        contra_arrays = np.stack(contra_arrays)

        classes_arrays_f = np.stack(classes_arrays_f)
        classes_arrays_b = np.stack(classes_arrays_b)
        classes_arrays = np.stack(classes_arrays)

        return {
            "greedy_deberta_entail_forward_full_trace": entail_arrays_f,
            "greedy_deberta_entail_backward_full_trace": entail_arrays_b,
            "greedy_deberta_entail_full_trace": entail_arrays,
            "greedy_deberta_contra_forward_full_trace": contra_arrays_f,
            "greedy_deberta_contra_backward_full_trace": contra_arrays_b,
            "greedy_deberta_contra_full_trace": contra_arrays,
            "greedy_deberta_classes_forward_full_trace": classes_arrays_f,
            "greedy_deberta_classes_backward_full_trace": classes_arrays_b,
            "greedy_deberta_classes_full_trace": classes_arrays,
        }
