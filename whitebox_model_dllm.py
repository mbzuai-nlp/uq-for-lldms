from lm_polygraph.utils.model import Model
from lm_polygraph.utils.generation_parameters import GenerationParameters
from transformers.generation import GenerateDecoderOnlyOutput
from typing import List, Dict, Optional, Union

import gc
import torch
import numpy as np
from typing import List
from llada.generate import (
    generate,
    generate_with_dual_cache,
    generate_with_prefix_cache,
    build_block_attention_mask,
)
from refusion.generate import generate_refusion
from torch.cuda import nvtx


class WhiteboxModelDiffusionLLM(Model):
    """Basic whitebox model adapter for using DiffusionLLM in stat calculators and uncertainty estimators."""

    def __init__(
        self,
        model,
        tokenizer,
        generation_parameters: GenerationParameters = GenerationParameters(),
        instruct: bool = False,
        device: str = "cuda",
        steps: int = 256,
        length: int = 128,
        block_length: int = 32,
        attention_device: str = "cuda",
        mask_id: int = 126336,
        use_cache: bool = True,
        prompt_suffix: str = None,
        model_path: Optional[str] = None,
        shift_logits: bool = False,
    ):
        self.prompt_suffix = prompt_suffix
        # Dream-7B is an AR-shifted diffusion model: the logit at position i
        # predicts token i+1. `shift_logits` (default False) enables the
        # shift-correction paths. For every other model it stays False and all
        # shift-gated branches below are no-ops, so behavior is identical to main.
        self.shift_logits = shift_logits
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generation_parameters = generation_parameters
        stop_strings = getattr(self.generation_parameters, "stop_strings", None)
        if stop_strings is None:
            stop_strings = []
        self.stop_sequences = stop_strings + [
            "<|role_end|>",
            "<|endoftext|>",
            "<|eot_id|>",
        ]
        num_hidden_layers = getattr(self.model.config, "num_hidden_layers", -1)
        self.hs_trace_layer = num_hidden_layers // 2 if num_hidden_layers > 0 else -1
        # Dream uses <|im_end|> as its chat turn terminator; add it only for
        # Dream so non-Dream models' stop handling is unchanged.
        if self.shift_logits:
            self.stop_sequences.append("<|im_end|>")
        self.stop_token_ids = [
            self.tokenizer.encode(stop_seq, add_special_tokens=False)
            for stop_seq in self.stop_sequences
        ]
        self.steps = steps
        self.length = length
        self.block_length = block_length
        self.mask_id = mask_id
        self.use_cache = use_cache
        self.base_device = device
        self.model_type = "CausalLM"
        self.instruct = instruct
        self.model_path = model_path

        self.attention_device = attention_device

        # LLaDA 2.1 (llada2_moe) requires explicit 4D attention masks
        model_cfg_type = getattr(self.model.config, "model_type", "")
        self.needs_attention_mask = model_cfg_type == "llada2_moe"

        if self.needs_attention_mask and self.use_cache:
            raise ValueError(
                "llada2_moe models require use_cache=False; "
                "the KV-cache generation path does not support explicit attention masks."
            )

        # LLaDA 2.1 chat template injects "detailed thinking off" as system message;
        # replace with a concise instruction for cleaner prompting
        if self.needs_attention_mask and self.tokenizer.chat_template:
            self.tokenizer.chat_template = self.tokenizer.chat_template.replace(
                "'detailed thinking ' + thinking_option",
                "'Answer concisely. Output only the requested format.'",
            )

    def generate(self, *args, **kwargs):

        do_sample = kwargs.get("do_sample", False)
        temperature = self.generation_parameters.temperature if do_sample else 0.0
        max_new_tokens = kwargs.get("max_new_tokens", self.length)

        def find_block_length(max_new_tokens, divisors=[4, 3, 2]):
            if max_new_tokens <= 32:
                return max_new_tokens
            for divisor in divisors:
                if max_new_tokens % divisor == 0:
                    return max_new_tokens // divisor
            return max_new_tokens

        # find the largest block_length that is a divisor of max_new_tokens
        # if max_new_tokens >= 128, the standard block_length is 32
        divisors = (
            np.arange(max_new_tokens // 32, 1, -1)
            if max_new_tokens >= 128
            else [4, 3, 2]
        )
        block_length = find_block_length(max_new_tokens, divisors=divisors)

        with torch.inference_mode():
            nvtx.range_push("INFER")
            if "refusion" in (self.model_path or "").lower():
                output = generate_refusion(
                    self.model,
                    self.tokenizer,
                    kwargs["input_ids"],
                    gen_length=max_new_tokens,
                    temperature=temperature,
                    mask_id=self.mask_id,
                    slot_size=4,
                    model_path=self.model_path or "",
                    serial_num_blocks=max(1, max_new_tokens // block_length),
                    slot_threshold=0.6,
                    token_threshold=0.3
                )
            elif self.use_cache:
                if self.shift_logits:
                    # Dream-style AR-shifted models: the dual-cache path reuses
                    # stale future-MASK KV from the warm-up forward and commits
                    # <|endoftext|> prematurely on summarization prompts (see
                    # scripts/validate_dream_cache.py). Route Dream to the
                    # prefix-cache path instead. This branch only runs for
                    # Dream (shift_logits=True); every other model takes the
                    # unchanged dual-cache path below.
                    output = generate_with_prefix_cache(
                        self.model,
                        kwargs["input_ids"],
                        steps=self.steps,
                        gen_length=max_new_tokens,
                        block_length=block_length,
                        temperature=temperature,
                        remasking="low_confidence",
                        threshold=0.9,
                        factor=1,
                        mask_id=self.mask_id,
                        tokenizer=self.tokenizer,
                        shift_logits=self.shift_logits,
                    )
                else:
                    output = generate_with_dual_cache(
                        self.model,
                        self.tokenizer,
                        kwargs["input_ids"],
                        steps=self.steps,
                        gen_length=max_new_tokens,
                        block_length=block_length,
                        temperature=temperature,
                        remasking="low_confidence",
                        threshold=0.9,
                        factor=1,
                        mask_id=self.mask_id,
                        output_hidden_states=(
                            kwargs.get("output_hidden_states_trace", False)
                            or kwargs.get("output_all_hidden_states_trace", False)
                        ),
                        hs_layer=self.hs_trace_layer,
                        output_all_hidden_states_trace=kwargs.get(
                            "output_all_hidden_states_trace", False
                        ),
                    )
            else:
                output = generate(
                    self.model,
                    kwargs["input_ids"],
                    steps=self.steps,
                    gen_length=max_new_tokens,
                    block_length=block_length,
                    temperature=temperature,
                    remasking="low_confidence",
                    threshold=0.9,
                    factor=1,
                    mask_id=self.mask_id,
                    tokenizer=self.tokenizer,
                    needs_attention_mask=self.needs_attention_mask,
                    shift_logits=self.shift_logits,
                )
            torch.cuda.synchronize()
            nvtx.range_pop()

        return self.post_processing(
            output,
            kwargs["input_ids"],
            min(1, kwargs["min_new_tokens"]),
            max_new_tokens,
            kwargs.get("output_attentions", False),
            kwargs.get("output_hidden_states", False),
            (
                kwargs.get("output_hidden_states_trace", False)
                or kwargs.get("output_all_hidden_states_trace", False)
            ),
        )

    def device(self):
        return self.base_device

    def tokenize(
        self, texts: Union[List[str], List[List[Dict[str, str]]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenizes input texts batch into a dictionary using the model tokenizer.

        Parameters:
            texts (List[str]): list of input texts batch.
        Returns:
            dict[str, torch.Tensor]: tensors dictionary obtained by tokenizing input texts batch.
        """
        # Apply chat template if tokenizer has it
        add_start_symbol = True
        if self.instruct:
            formatted_texts = []
            for chat in texts:
                if isinstance(chat, str):
                    if self.prompt_suffix:
                        last_answer = chat.rfind("Answer:")
                        if last_answer != -1:
                            chat = chat[:last_answer] + self.prompt_suffix + "\n" + chat[last_answer:]
                        else:
                            chat = chat + "\n" + self.prompt_suffix
                    chat = [{"role": "user", "content": chat}]
                formatted_chat = self.tokenizer.apply_chat_template(
                    chat, add_generation_prompt=True, tokenize=False
                )
                formatted_texts.append(formatted_chat)
            texts = formatted_texts

            add_start_symbol = False
        return self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=add_start_symbol,
            return_token_type_ids=False,
        )

    def __call__(self, **args):
        """
        Calls the model on the input batch. Returns the resulted scores.
        """
        return self.model(**args)

    def _legacy_generate_texts(self, *args, **kwargs):
        outputs = self.generate(*args, **kwargs)
        texts = [
            self.tokenizer.decode(outputs, skip_special_tokens=True)
            for outputs in outputs.sequences
        ]
        return texts

    def _is_text_generation_input(self, input_texts):
        if isinstance(input_texts, str):
            return True
        if not isinstance(input_texts, (list, tuple)):
            return False
        return all(
            isinstance(text, str)
            or (
                isinstance(text, list)
                and all(isinstance(message, dict) for message in text)
            )
            for text in input_texts
        )

    def generate_texts(self, *args, **kwargs):
        input_texts = None
        generation_kwargs = kwargs.copy()

        if "input_texts" in generation_kwargs and len(args) == 0:
            input_texts = generation_kwargs.pop("input_texts")
        elif len(args) == 1 and "input_texts" not in generation_kwargs:
            input_texts = args[0]
        else:
            return self._legacy_generate_texts(*args, **kwargs)

        if not self._is_text_generation_input(input_texts):
            return self._legacy_generate_texts(*args, **kwargs)

        if isinstance(input_texts, str):
            input_texts = [input_texts]
        else:
            input_texts = list(input_texts)

        num_return_sequences = int(
            generation_kwargs.pop("num_return_sequences", 1) or 1
        )
        if "min_new_tokens" not in generation_kwargs:
            generation_kwargs["min_new_tokens"] = generation_kwargs.pop(
                "min_length", 1
            )
        else:
            generation_kwargs.pop("min_length", None)

        if num_return_sequences > 1:
            input_texts = [
                input_text
                for input_text in input_texts
                for _ in range(num_return_sequences)
            ]

        batch: Dict[str, torch.Tensor] = self.tokenize(input_texts)
        batch = {key: value.to(self.device()) for key, value in batch.items()}
        sequences = self.generate(**batch, **generation_kwargs).sequences.cpu()
        input_len = batch["input_ids"].shape[1]

        return [
            self.tokenizer.decode(sequence[input_len:], skip_special_tokens=True)
            for sequence in sequences
        ]

    def rearrange_hidden_states(self, hidden_states, prompt_len, token_len):
        """
        hidden_states: tuple of num_layers tensors
                    each tensor shape [1, prompt_len + token_len, embedding_dim]

        returns: tuple of length token_len
                each element is a tuple of num_layers tensors
        """
        result = []
        for t in range(token_len):
            per_token = []

            for layer in hidden_states:
                if t == 0:
                    # prompt tensor only once per token group
                    data = torch.nan_to_num(
                        layer[:, :prompt_len, :], nan=0.0, posinf=0.0, neginf=0.0
                    )
                    per_token.append(data)
                else:
                    data = torch.nan_to_num(
                        layer[:, prompt_len + t - 1 : prompt_len + t, :],
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    per_token.append(data)

            result.append(tuple(per_token))
        return tuple(result)

    def process_attention(self, attentions, token_idx, prompt_length):
        """
        Process the attention tensors for a specific token index and prompt length.

        Args:
            attentions (list[Tensor]): List of attention tensors.
            token_idx (int): Index of the current token.
            prompt_length (int): Length of the prompt.

        Returns:
            Tensor: Processed attention tensor with NaNs and infs replaced by 0.0.
        """
        same_device = self.base_device == self.attention_device

        if token_idx == 0:
            if same_device:
                data = torch.cat(
                    [
                        attention[:, :, :prompt_length, :prompt_length]
                        for attention in attentions
                    ]
                ).unsqueeze(-4)
            else:
                shape = attentions[0][:, :, :prompt_length, :prompt_length].shape[1:]
                data = torch.empty(
                    (len(attentions), *shape),
                    device="cpu",
                    dtype=attentions[0].dtype,
                    pin_memory=True,
                )
                for i, attention in enumerate(attentions):
                    data.narrow(0, i, 1).copy_(
                        attention[:, :, :prompt_length, :prompt_length],
                        non_blocking=True,
                    )
                data.unsqueeze_(-4)
        else:
            start = prompt_length + token_idx - 1
            end = prompt_length + token_idx
            if same_device:
                data = (
                    torch.cat(
                        [attention[:, :, start, :end] for attention in attentions]
                    )
                    .unsqueeze(-3)
                    .unsqueeze(-2)
                )
            else:
                shape = attentions[0][:, :, start, :end].shape[1:]
                data = torch.empty(
                    (len(attentions), *shape),
                    device="cpu",
                    dtype=attentions[0].dtype,
                    pin_memory=True,
                )
                for i, attention in enumerate(attentions):
                    data.narrow(0, i, 1).copy_(
                        attention[:, :, start, :end], non_blocking=True
                    )
                data.unsqueeze_(-3).unsqueeze_(-2)

        data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        return data

    def post_processing(
        self,
        outputs,
        input_ids,
        min_new_tokens,
        max_new_tokens,
        output_attentions=False,
        output_hidden_states=False,
        output_hidden_states_trace=False,
    ):
        sequences = outputs[0]

        # Truncate sequences to reasonable length to avoid OOM
        # Keep prompt + max_new_tokens, but cap at 2048 tokens for memory efficiency
        prompt_length = input_ids.shape[1]
        max_sequence_length = 2048
        max_allowed_length = min(prompt_length + max_new_tokens, max_sequence_length)

        if sequences.shape[1] > max_allowed_length:
            # Truncate from the left to keep the most recent tokens
            excess = sequences.shape[1] - max_allowed_length
            sequences = sequences[:, excess:]
            # Adjust prompt_length after truncation from the left
            prompt_length = max(0, prompt_length - excess)

        with torch.inference_mode():
            model_kwargs = {}
            if self.needs_attention_mask:
                model_kwargs["attention_mask"] = build_block_attention_mask(
                    sequences.shape[0], sequences.shape[1], sequences.device
                )
            model_output = self.model(
                sequences,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                **model_kwargs,
            )

        # Dream-style AR-shifted models: the logit at position i predicts token
        # i+1, so shift right by one to align logits with the tokens they score.
        # No-op for every other model (shift_logits=False).
        if self.shift_logits:
            model_output.logits = torch.cat(
                [model_output.logits[:, :1], model_output.logits[:, :-1]], dim=1
            )

        # Move to CPU and clear cache to free GPU memory
        logits = model_output.logits.log_softmax(dim=-1).to(torch.float16).cpu()
        attentions = model_output.attentions
        # prompt_length = input_ids.shape[1]

        processed_sequences = []
        processed_logits = []

        for i in range(sequences.shape[0]):
            generated_token_ids = sequences[i, prompt_length:]
            trunc_length = generated_token_ids.shape[0]
            if max_new_tokens > 1:
                # Find first stop sequence in token ids (earliest among all)
                earliest_stop = generated_token_ids.shape[0]
                for stop_seq_ids in self.stop_token_ids:
                    if not stop_seq_ids:
                        continue
                    stop_seq_len = len(stop_seq_ids)
                    if stop_seq_len == 0 or stop_seq_len > trunc_length:
                        continue
                    for idx in range(min_new_tokens, trunc_length - stop_seq_len + 1):
                        if torch.equal(
                            generated_token_ids[idx : idx + stop_seq_len],
                            torch.tensor(
                                stop_seq_ids, device=generated_token_ids.device
                            ),
                        ):
                            earliest_stop = min(earliest_stop, idx)
                            break
                trunc_length = earliest_stop

            # Dream-only: text-level stop check catches stop strings that BPE
            # merged across token boundaries (so the token-level scan above
            # missed them). Gated on shift_logits so non-Dream truncation is
            # unchanged.
            if self.shift_logits and max_new_tokens > 1 and self.stop_sequences:
                text = self.tokenizer.decode(
                    generated_token_ids[:trunc_length], skip_special_tokens=True
                )
                earliest_text_stop = len(text)
                for stop_seq in self.stop_sequences:
                    idx = text.find(stop_seq, 1)  # skip first char
                    if idx > 0:
                        earliest_text_stop = min(earliest_text_stop, idx)
                if earliest_text_stop < len(text):
                    truncated_text = text[:earliest_text_stop]
                    new_ids = self.tokenizer.encode(
                        truncated_text, add_special_tokens=False
                    )
                    trunc_length = min(trunc_length, len(new_ids))

            trimmed_generated = generated_token_ids[:trunc_length].to(input_ids[i].device)
            n_tokens = trimmed_generated.shape[0]
            processed_sequences.append(torch.cat([input_ids[i], trimmed_generated]))
            processed_logits.append(
                logits[i, prompt_length : prompt_length + n_tokens, :]
            )

        hidden_states = None
        if output_hidden_states:
            hidden_states = self.rearrange_hidden_states(
                model_output.hidden_states, prompt_length, n_tokens
            )
        processed_sequences = torch.stack(processed_sequences, dim=0).cpu()
        processed_logits = torch.stack(processed_logits, dim=0).transpose(1, 0).cpu()
        processed_logits = tuple(logit for logit in processed_logits)

        attentions_greedy = []
        has_attentions = (
            output_attentions
            and attentions is not None
            and len(attentions) > 0
            and attentions[0] is not None
        )
        if has_attentions:
            for token_idx in range(n_tokens):
                data = self.process_attention(attentions, token_idx, prompt_length)
                attentions_greedy.append(data)

        standard_output = GenerateDecoderOnlyOutput(
            sequences=processed_sequences,
            logits=processed_logits,
            scores=processed_logits,
            attentions=attentions_greedy,
            hidden_states=hidden_states,
        )
        nfe = outputs[1]
        if isinstance(nfe, int):
            nfe = np.array([nfe])
        standard_output.nfe = nfe
        standard_output.remask_count = outputs[2]["remask_count"].cpu().detach()
        standard_output.first_remask_step = (
            outputs[2]["first_remask_step"][:, prompt_length:].cpu().detach()
        )
        generation_trace = [
            texts for texts in outputs[2]["generation_trace"] if len(texts)
        ]
        generation_trace = list(map(list, zip(*generation_trace)))
        standard_output.generation_trace = generation_trace

        full_generation_trace = [
            texts for texts in outputs[2]["full_generation_trace"] if len(texts)
        ]
        full_generation_trace = list(map(list, zip(*full_generation_trace)))
        standard_output.full_generation_trace = full_generation_trace

        last_block_generation_trace = [
            texts for texts in outputs[2]["last_block_generation_trace"] if len(texts)
        ]
        last_block_generation_trace = list(map(list, zip(*last_block_generation_trace)))
        standard_output.last_block_generation_trace = last_block_generation_trace

        # Per-block traces without concatenation - preserve block structure
        # block_generation_traces: list of blocks [block][step][bs]
        # We want: list of batch items [bs][block][step]
        block_generation_traces = outputs[2]["block_generation_traces"]
        # Transpose each block: [block][step][bs] -> [block][bs][step]
        block_generation_traces = [
            [list(step_strings) for step_strings in zip(*block_trace)] if block_trace else []
            for block_trace in block_generation_traces
        ]
        # Filter out empty blocks
        block_generation_traces = [block_trace for block_trace in block_generation_traces if block_trace]
        # Transpose to get batch first: [block][bs][step] -> [bs][block][step]
        if block_generation_traces:
            block_generation_traces = [list(batch_blocks) for batch_blocks in zip(*block_generation_traces)]
        standard_output.block_generation_traces = block_generation_traces

        # Add gen_trace_logits and gen_trace_entropy from outputs[2]
        standard_output.gen_trace_logits = outputs[2]["gen_trace_logits"].cpu().detach()
        standard_output.gen_trace_entropy = outputs[2]["gen_trace_entropy"].cpu().detach()
        standard_output.gen_trace_argmax = outputs[2]["gen_trace_argmax"].cpu().detach()
        standard_output.gen_trace_top2gap = outputs[2]["gen_trace_top2gap"].cpu().detach()
        standard_output.confidence_at_commit = outputs[2]["confidence_at_commit"][:, prompt_length:].cpu().detach()
        standard_output.mask_fraction_at_commit = outputs[2]["mask_fraction_at_commit"][:, prompt_length:].cpu().detach()

        if output_hidden_states_trace:
            hidden_states_trace = outputs[2]["hidden_states_trace"]
            hidden_states_trace_mask = outputs[2]["hidden_states_trace_mask"]
            standard_output.hidden_states_trace = hidden_states_trace
            standard_output.hidden_states_trace_mask = hidden_states_trace_mask
            standard_output.hidden_states_trace_block_length = outputs[2][
                "hidden_states_trace_block_length"
            ]

        standard_output.mask_id = self.mask_id

        # Clean up GPU memory
        del model_output
        torch.cuda.empty_cache()
        gc.collect()

        return standard_output
