import torch
import numpy as np

from typing import Dict, List, Tuple, Union

from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from whitebox_model_dllm import WhiteboxModelDiffusionLLM

import torch.nn.functional as F
from .mc_masked_ll_calculator import get_log_likelihood


class SampleMCMaskedLLCalculator(StatCalculator):
    """
    For Whitebox model (lm_polygraph.WhiteboxModel), at input texts batch calculates:
    * MC masked log likelihoods for each sample in sample_texts.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """
        return [
            "sample_mc_masked_ll",
        ], ["sample_texts"]

    def __init__(self, mc_num: int = 16, seed: int = None):
        super().__init__()
        self.mc_num = mc_num
        self.seed = seed

    def __call__(
        self,
        dependencies: Dict[str, np.ndarray],
        texts: List[str],
        model: Union[WhiteboxModelDiffusionLLM],
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the MC masked log-likelihoods (SampleMCMaskedLLCalculator) for each sample in the batch.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, containing model, tokenizer, sample_texts, and input_texts.
            texts (List[str]): Input texts batch used for model generation (ignored here).
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with
                - 'sample_mc_masked_ll' (List[List[float]]): MC masked log likelihood for each sample.
        """
        model = dependencies["model"]
        tokenizer = dependencies["tokenizer"]
        sample_texts = dependencies["sample_texts"]
        input_texts = dependencies["input_texts"]
        mask_id = dependencies.get("mask_id", 126336)

        sample_mc_masked_ll = []
        sample_mc_masked_normalized_ll = []
        for texts, input_text in zip(sample_texts, input_texts):
            sample_mc_masked_ll.append([])
            sample_mc_masked_normalized_ll.append([])
            for text in texts:
                prompt = torch.tensor(tokenizer(input_text)["input_ids"]).to(
                    model.device()
                )
                answer = torch.tensor(tokenizer(text)["input_ids"]).to(model.device())
                if len(answer):
                    mc_masked_ll_i, mc_masked_normalized_ll_i = get_log_likelihood(
                        model.model,
                        prompt,
                        answer,
                        mc_num=self.mc_num,
                        mask_id=mask_id,
                        seed=self.seed if self.seed is not None else None,
                    )
                    sample_mc_masked_ll[-1].append(mc_masked_ll_i)
                    sample_mc_masked_normalized_ll[-1].append(mc_masked_normalized_ll_i)
                else:
                    sample_mc_masked_ll[-1].append(1e3)
                    sample_mc_masked_normalized_ll[-1].append(1e3)

        result_dict = {
            "sample_mc_masked_ll": np.array(sample_mc_masked_ll),
            "sample_mc_masked_normalized_ll": np.array(sample_mc_masked_normalized_ll),
        }

        return result_dict
