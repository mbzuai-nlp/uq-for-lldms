import torch
import numpy as np

from typing import Dict, List, Tuple, Union

from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.model_adapters import WhiteboxModel, WhiteboxModelvLLM


class DiffusionGreedyProbsCalculator(StatCalculator):
    """
    For Whitebox model (lm_polygraph.WhiteboxModel), at input texts batch calculates:
    * generation texts
    * tokens of the generation texts
    * probabilities distribution of the generated tokens
    * attention masks across the model (if applicable)
    * embeddings from the model
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """
        return [
            "input_texts",
            "input_tokens",
            "greedy_log_probs",
            "greedy_tokens",
            "greedy_tokens_alternatives",
            "greedy_texts",
            "greedy_log_likelihoods",
            "input_embeddings_mean",
            "gen_trace_embeddings",
            "embeddings_raw",
            "embeddings_decoder_raw",
            "layers",
            "attention_raw",
            "tokenizer",
            "nfe",
            "remask_count",
            "first_remask_step",
            "generation_trace",
            "full_generation_trace",
            "last_block_generation_trace",
            "block_generation_traces",
            "gen_trace_logits",
            "gen_trace_entropy",
            "gen_trace_argmax",
            "gen_trace_top2gap",
            "confidence_at_commit",
            "mask_fraction_at_commit",
            "hidden_states_trace",
            "hidden_states_trace_mask",
            "hidden_states_trace_block_length",
        ], []

    def __init__(
        self,
        output_attentions: bool = True,
        output_hidden_states: bool = False,
        output_diffusion_stats: bool = False,
        output_token_embeddings: bool = False,
        hidden_state_layer=None,
        output_hidden_states_trace: bool = False,
        output_all_hidden_states_trace: bool = False,
        n_alternatives: int = 10,
    ):
        super().__init__()
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states or output_token_embeddings
        self.output_diffusion_stats = output_diffusion_stats
        self.output_token_embeddings = output_token_embeddings
        self.hidden_state_layer = hidden_state_layer
        self.output_hidden_states_trace = output_hidden_states_trace
        self.output_all_hidden_states_trace = output_all_hidden_states_trace
        self.n_alternatives = n_alternatives

    def _resolve_hidden_state_layer(self, n_layers):
        if self.hidden_state_layer is None or self.hidden_state_layer == "middle":
            return n_layers // 2
        layer = int(self.hidden_state_layer)
        if layer < 0:
            layer = n_layers + layer
        return max(0, min(layer, n_layers - 1))

    def _decoder_layer_states(self, hidden_states):
        if hidden_states is None or len(hidden_states) == 0:
            raise ValueError("output_token_embeddings=True requires embeddings_decoder_raw")
        first = hidden_states[0]
        if isinstance(first, (tuple, list)):
            layer = self._resolve_hidden_state_layer(len(first))
            prompt_states = first[layer].float()
            gen_parts = [step[layer].float() for step in hidden_states[1:]]
            if gen_parts:
                gen_states = torch.cat(gen_parts, dim=1)
            else:
                gen_states = prompt_states.new_zeros(
                    prompt_states.shape[0], 0, prompt_states.shape[-1]
                )
            return prompt_states, gen_states

        layer = self._resolve_hidden_state_layer(len(hidden_states))
        states = hidden_states[layer].float()
        return states, states.new_zeros(states.shape[0], 0, states.shape[-1])

    def _mean_decoder_embeddings(self, hidden_states, attention_mask):
        prompt_states, gen_states = self._decoder_layer_states(hidden_states)
        mask = attention_mask[:, : prompt_states.shape[1]].to(prompt_states.device)
        mask = mask.unsqueeze(-1).float()
        input_mean = (prompt_states * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        if gen_states.shape[1] == 0:
            gen_mean = torch.zeros_like(input_mean)
        else:
            gen_mean = gen_states.mean(1)
        return input_mean.to(torch.float16).cpu(), gen_mean.to(torch.float16).cpu()

    def _preprocess_attention(
        self,
        attentions: torch.Tensor,
        current_idx: int,
        start_idx: int,
        end_idx: int,
        prompt_len: int,
    ) -> torch.Tensor:
        """
        Preprocess attention weights before stacking.

        Parameters:
            attentions (torch.Tensor): Attention weights from a specific layer and head for a current token
            current_idx (int): Current position in the sequence
            start_idx (int): Start index of the generated tokens
            end_idx (int): End index of the generated tokens for current position
            prompt_len (int): Length of the prompt

        Returns:
            torch.Tensor: Preprocessed attention weights
        """
        # Handle attention tensor processing for models with varying attention sizes (e.g. Gemma)
        n_attentions = attentions.shape[-1]

        # Handle empty tensor case
        if attentions.nelement() == 0:
            return torch.zeros(abs(current_idx), device=attentions.device)

        # Handle cases where attention size is smaller than expected
        if n_attentions < end_idx:
            if start_idx < 0:
                return attentions[start_idx:n_attentions]
            return attentions[n_attentions - current_idx : n_attentions]

        # Handle cases where attention spans beyond expected range
        if (n_attentions - current_idx) > end_idx and start_idx < 0:
            return attentions[prompt_len : prompt_len + current_idx]

        # Default case: return attention slice within expected range
        return attentions[start_idx:end_idx]

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: Union[WhiteboxModel, WhiteboxModelvLLM],
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the statistics of probabilities at each token position in the generation.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'input_tokens' (List[List[int]]): tokenized input texts,
                - 'greedy_log_probs' (List[List[np.array]]): logarithms of autoregressive
                        probability distributions at each token,
                - 'greedy_texts' (List[str]): model generations corresponding to the inputs,
                - 'greedy_tokens' (List[List[int]]): tokenized model generations,
                - 'attention' (List[List[np.array]]): attention maps at each token, if applicable to the model,
                - 'greedy_log_likelihoods' (List[List[float]]): log-probabilities of the generated tokens.
        """
        if dependencies.get("layers") is None:
            dependencies["layers"] = [-1]
        else:
            dependencies["layers"] = [
                -1 if layer is None else layer for layer in dependencies["layers"]
            ]

        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        with torch.no_grad():
            out = model.generate(
                **batch,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                output_attentions=self.output_attentions,
                output_hidden_states=self.output_hidden_states,
                output_hidden_states_trace=self.output_hidden_states_trace,
                output_all_hidden_states_trace=self.output_all_hidden_states_trace,
                num_return_sequences=1,
                suppress_tokens=(
                    []
                    if model.generation_parameters.allow_newlines
                    else [
                        t
                        for t in range(len(model.tokenizer))
                        if "\n" in model.tokenizer.decode([t])
                    ]
                ),
            )
            logits = torch.stack(out.scores, dim=1)
            if model.model_type == "vLLMCausalLM":
                logits = logits.transpose(1, 0)
            sequences = out.sequences
            if self.output_attentions:
                attentions = out.attentions
            if not self.output_hidden_states:
                embeddings_dict = {}
            elif model.model_type in ["CausalLM", "VisualLM"]:
                embeddings_dict = {
                    "embeddings_decoder_raw": out.hidden_states,
                }
                if model.model_type == "VisualLM":
                    embeddings_dict["embeddings_visual_raw"] = out.vision_hidden_states
            elif model.model_type == "Seq2SeqLM":
                embeddings_dict = {
                    "embeddings_encoder_raw": out.encoder_hidden_states,
                    "embeddings_decoder_raw": out.decoder_hidden_states,
                }
            else:
                raise NotImplementedError

        cut_logits = []
        cut_sequences = []
        cut_texts = []
        cut_alternatives = []
        for i in range(len(texts)):
            if model.model_type == "CausalLM":
                idx = batch["input_ids"].shape[1]
                seq = sequences[i, idx:].cpu()
            elif model.model_type == "vLLMCausalLM":
                seq = sequences[i].cpu()
            else:
                seq = sequences[i, 1:].cpu()
            length, text_length = len(seq), len(seq)
            for j in range(len(seq)):
                if seq[j] == model.tokenizer.eos_token_id:
                    length = j + 1
                    text_length = j
                    break
            cut_sequences.append(seq[:length].tolist())
            cut_texts.append(model.tokenizer.decode(seq[:text_length]))
            cut_logits.append(logits[i, :length, :].cpu().numpy())
            cut_alternatives.append([[] for _ in range(length)])
            for j in range(length):
                lt = logits[i, j, :].cpu().numpy()
                best_tokens = np.argpartition(lt, -self.n_alternatives)
                ln = len(best_tokens)
                best_tokens = best_tokens[ln - self.n_alternatives : ln]
                for t in best_tokens:
                    cut_alternatives[-1][j].append((t.item(), lt[t].item()))
                cut_alternatives[-1][j].sort(
                    key=lambda x: x[0] == cut_sequences[-1][j],
                    reverse=True,
                )

        ll = []
        for i in range(len(texts)):
            log_probs = cut_logits[i]
            tokens = cut_sequences[i]
            assert len(tokens) == len(log_probs)
            ll.append([log_probs[j, tokens[j]] for j in range(len(log_probs))])

        result_dict = {
            "input_tokens": batch["input_ids"].to("cpu").tolist(),
            "greedy_log_probs": cut_logits,
            "greedy_tokens": cut_sequences,
            "greedy_tokens_alternatives": cut_alternatives,
            "greedy_texts": cut_texts,
            "greedy_log_likelihoods": ll,
        }
        result_dict.update(embeddings_dict)
        if self.output_hidden_states:
            result_dict["layers"] = [-1]
        if self.output_token_embeddings:
            input_embeddings, gen_embeddings = self._mean_decoder_embeddings(
                embeddings_dict.get("embeddings_decoder_raw"),
                batch.get("attention_mask", torch.ones_like(batch["input_ids"])),
            )
            result_dict["input_embeddings_mean"] = input_embeddings
            result_dict["gen_trace_embeddings"] = gen_embeddings
        result_dict.update({"tokenizer": model.tokenizer})

        if self.output_attentions:
            result_dict.update({"attention_raw": attentions})

        if self.output_diffusion_stats:
            result_dict.update({"nfe": out.nfe})
            result_dict.update({"remask_count": out.remask_count})
            result_dict.update({"first_remask_step": out.first_remask_step})
            result_dict.update({"generation_trace": out.generation_trace})
            result_dict.update({"full_generation_trace": out.full_generation_trace})
            result_dict.update({"last_block_generation_trace": out.last_block_generation_trace})
            result_dict.update({"block_generation_traces": out.block_generation_traces})
            result_dict.update({"gen_trace_logits": out.gen_trace_logits})
            result_dict.update({"gen_trace_entropy": out.gen_trace_entropy})
            result_dict.update({"gen_trace_argmax": out.gen_trace_argmax})
            result_dict.update({"gen_trace_top2gap": out.gen_trace_top2gap})
            result_dict.update({"confidence_at_commit": out.confidence_at_commit})
            result_dict.update({"mask_fraction_at_commit": out.mask_fraction_at_commit})
            if self.output_hidden_states_trace or self.output_all_hidden_states_trace:
                result_dict.update({"hidden_states_trace": out.hidden_states_trace})
                result_dict.update({"hidden_states_trace_mask": out.hidden_states_trace_mask})
                result_dict.update(
                    {
                        "hidden_states_trace_block_length": [
                            out.hidden_states_trace_block_length
                        ]
                        * len(texts)
                    }
                )
        if hasattr(out, "mask_id"):
            result_dict["mask_id"] = out.mask_id

        return result_dict
