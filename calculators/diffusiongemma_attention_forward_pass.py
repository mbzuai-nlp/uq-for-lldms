import types

from lm_polygraph.stat_calculators.attention_forward_pass import AttentionForwardPassCalculator


class DiffusionGemmaAttentionForwardPassCalculator(AttentionForwardPassCalculator):
    """lm-polygraph's AttentionForwardPassCalculator for DiffusionGemma.

    The built-in calls ``model.model(input_ids, output_attentions=True)`` on the HF model; here
    that call is routed to the backend's ``__call__`` (causal encoder pass over prompt + answer),
    everything else is the library code.
    """

    def __call__(self, dependencies, texts, model, max_new_tokens: int = 100):
        proxy = types.SimpleNamespace(model=model, tokenize=model.tokenize, device=model.device)
        return super().__call__(dependencies, texts, proxy, max_new_tokens)
