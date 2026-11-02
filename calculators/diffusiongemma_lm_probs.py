import numpy as np
import torch

from typing import Dict, List, Tuple

from lm_polygraph.stat_calculators.stat_calculator import StatCalculator


class DiffusionGemmaGreedyLMProbsCalculator(StatCalculator):
    """DiffusionGemma counterpart of lm-polygraph's GreedyLMProbsCalculator (PMI-based estimators).

    The built-in calculator scores the answer with ``model.model(**batch).logits`` as an
    autoregressive LM, which DiffusionGemma's encoder-decoder diffusion model cannot serve. Here
    log p(y) is taken from the backend's clean pass (``_final_forward``) without input context:
    the prompt is an empty user turn and the canvas is the reply as the model emits it
    ("<|channel>thought\\n<channel|>" + answer + "<turn|>"), padded to the canvas length.
    Output format matches the built-in: ``greedy_lm_log_probs`` without the last token and
    ``greedy_lm_log_likelihoods`` for every answer token.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return ["greedy_lm_log_probs", "greedy_lm_log_likelihoods"], ["greedy_tokens"]

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        tok = model.tokenizer
        prompt = {k: v.to(model.device()) for k, v in model.tokenize([""]).items()}
        prefix = tok.encode("<|channel>thought\n<channel|>", add_special_tokens=False)
        end_of_turn = tok.convert_tokens_to_ids("<turn|>")
        canvas_length = model.model.config.canvas_length

        greedy_lm_log_probs, greedy_lm_ll = [], []
        for tokens in dependencies["greedy_tokens"]:
            canvas = (prefix + list(tokens) + [end_of_turn])[:canvas_length]
            canvas += [tok.pad_token_id] * (canvas_length - len(canvas))
            canvas = torch.tensor([canvas], device=model.device())
            logp, _, _ = model._final_forward(prompt["input_ids"], prompt.get("attention_mask"), canvas)
            logp = logp[0, len(prefix) : len(prefix) + len(tokens)].float()
            greedy_lm_log_probs.append(logp[:-1].cpu().numpy())
            greedy_lm_ll.append([logp[j, t].item() for j, t in enumerate(tokens)])

        return {
            "greedy_lm_log_probs": greedy_lm_log_probs,
            "greedy_lm_log_likelihoods": greedy_lm_ll,
        }
