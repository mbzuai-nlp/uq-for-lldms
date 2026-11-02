"""Whitebox UQ backend for Google DiffusionGemma (``google/diffusiongemma-26B-A4B-it``).

DiffusionGemma is an encoder-decoder BLOCK/canvas text-diffusion LM with UNIFORM
corruption (no persistent mask token) — architecturally different from masked
diffusion (LLaDA/Dream). ``generate()`` returns ``scores=None`` and does NOT accept a
``use_cache`` argument (the KV cache is always on). See
``docs/diffusiongemma/{00_architecture,01_harness_contract,02_mechanics}.md``.

It drives the OFFICIAL DiffusionGemma ``generate()`` with two read-only,
algorithm-preserving hooks — a logits processor (:class:`UQTraceProcessor`) and a draft
streamer (:class:`UQTraceStreamer`) — then returns the same output object as the LLaDA
backend (:class:`WhiteboxModelDiffusionLLM`): sequences, per-token ``scores``, attentions and
hidden states from one clean pass over the final canvas, plus the diffusion trace stats. The
LLaDA calculators (greedy probs, sampling, attention, embeddings, training statistics) therefore
run unchanged (``model_type == "CausalLM"``).

Two load-bearing design decisions (docs/diffusiongemma/00_architecture.md):
  1. The output attribute ``greedy_log_likelihoods`` (read by the DiffusionGemma-specific
     calculators) is the per-answer-position log-prob of each committed token at its commit
     step; ``scores`` (read by the LLaDA calculators) come from the final clean pass.
  2. ``confidence_at_commit`` is redefined via argmax-stabilisation (acceptance is
     non-monotonic and there is no persistent mask): a position's commit step is the
     last step at which its argmax still disagreed with the final token, and the
     confidence is ``log P(argmax)`` at that step.

Every assumption that can only be checked against the real model/GPU is marked
``# VERIFY@runtime`` and exercised by ``scripts/validate_diffusiongemma.py``.
"""
from __future__ import annotations

import contextlib
import gc
import types
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from lm_polygraph.utils.model import Model
from lm_polygraph.utils.generation_parameters import GenerationParameters
from transformers.generation import GenerateDecoderOnlyOutput

from whitebox_model_dllm import WhiteboxModelDiffusionLLM


@contextlib.contextmanager
def _eager_attention(model, enabled: bool):
    """Switch to eager attention while attention weights are requested (sdpa does not return them)."""
    prev = model.config._attn_implementation
    if enabled:
        model.set_attn_implementation("eager")
    try:
        yield
    finally:
        if enabled:
            model.set_attn_implementation(prev)


# ---------------------------------------------------------------------------
# Read-only instrumentation hooks.
# Copied VERBATIM from scripts/validate_diffusiongemma.py so the backend uses the
# exact processor/streamer/commit logic that the runtime contract-verifier checks.
# ---------------------------------------------------------------------------
class UQTraceProcessor:
    """Read-only logits processor. Reduces (B, C, V) -> four (B, C) tensors/step.

    MUST return `scores` unchanged (the same tensor feeds acceptance + self-conditioning).
    """

    def __init__(self):
        self.entropy, self.maxlp, self.top2gap, self.argmax = [], [], [], []
        self.nfe = 0
        self.seen_kwargs = None
        self.scores_shape = None

    def __call__(self, input_ids, scores, cur_step=None):
        # transformers 5.x validates the signature: params beyond (input_ids, scores)
        # must be keys the model actually passes. DiffusionGemma passes only cur_step
        # (generation_diffusion_gemma.py: logits_processor(input_ids, raw_logits, cur_step=...)).
        if self.scores_shape is None:
            self.seen_kwargs = ["cur_step"] if cur_step is not None else []
            self.scores_shape = tuple(scores.shape)
        self.nfe += 1
        s = scores.float()
        lp = s.log_softmax(-1)
        p = lp.exp()
        self.entropy.append((-(p * lp).sum(-1)).to(torch.float16).cpu())
        self.maxlp.append(lp.max(-1).values.to(torch.float16).cpu())
        t2 = p.topk(2, -1).values
        self.top2gap.append((t2[..., 0] - t2[..., 1]).to(torch.float16).cpu())
        self.argmax.append(scores.argmax(-1).to(torch.int32).cpu())
        # Last step's raw logits: the self-conditioning input of the final forward pass.
        self.last_scores, self.last_step = scores, cur_step
        return scores  # IDENTITY — never mutate


class GumbelNoiseLogitsProcessor:
    """Optional temperature knob for the decode. Not required for sampling to work.

    DiffusionGemma is already stochastic: the canvas is initialised with random tokens and
    non-accepted positions are renoised, so repeated decodes of the same prompt differ
    whenever the model is genuinely uncertain. Measured with no processor attached: 5/5
    distinct continuations on an open-ended prompt, 3/5 on an obscure date, and 1/5 on
    near-certain facts such as the capital of Australia. The last case is correct calibrated
    behaviour rather than determinism, and it is easy to mistake for a broken sampler if the
    only prompts tried are ones the model is sure about.

    This processor exists to *widen* that distribution on demand. The sampler commits
    ``argmax(logits_processor(raw_logits))``, and by the Gumbel-max trick
    ``argmax(logits / T + G)`` with ``G ~ Gumbel(0, 1)`` is an exact sample from
    ``softmax(logits / T)``, so attaching it converts the argmax into a temperature-``T``
    draw. Measured share of argmax decisions changed: 1.0% at T=0.5, 6.8% at T=1.0, and
    96.6% at T=2.0, where the output degenerates into word salad. Useful for a
    diversity ablation; leave it off for normal runs.

    Register it AFTER :class:`UQTraceProcessor` so the trace statistics still see raw,
    un-noised logits. Note that samples drawn with it come from a different decoding
    distribution than the greedy output, whereas native sampling does not have that caveat.

    KNOWN LIMITATION. Because the trace is captured before the noise is added, the recorded
    argmaxes and max log-probabilities describe the un-noised distribution while the tokens
    actually committed come from the noised one. ``commit_analysis`` then compares raw
    argmaxes against noised final tokens, and the synthesized ``greedy_log_likelihoods`` hold
    the raw maximum rather than the probability of the token that was committed. So with a
    temperature set, the likelihood-derived statistics (MSP, the sample log-probs in
    ``DiffusionGemmaSamplingGenerationCalculator``) are not trustworthy, even though the
    generated text is. All reported runs use ``sample_temperature=None`` and are unaffected.
    Fixing this properly means capturing the trace after the noise, which would make the
    trajectory statistics describe the noised distribution instead; leave it until a run
    actually needs the temperature knob.
    """

    def __init__(self, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0 (got {temperature})")
        self.temperature = float(temperature)

    def __call__(self, input_ids, scores, cur_step=None):
        # float32 for the noise: bf16 has too few mantissa bits for a usable Gumbel draw.
        # This allocates one extra float32 tensor the size of the logits, which is why the
        # sampling sweeps run at batch_size=1.
        s = scores.float() / self.temperature
        u = torch.rand_like(s).clamp_(1e-20, 1.0)
        g = -torch.log(-torch.log(u))
        return s + g


class UQTraceStreamer:
    _takes_logits = False

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.drafts = []
        self.put_draft_arg = None

    def put_draft(self, value=None, logits=None, **kwargs):
        if self.put_draft_arg is None:
            self.put_draft_arg = "value" if value is not None else ("logits" if logits is not None else "?")
        # Keep the raw argmax canvas ids (B, C); they are decoded in _assemble_output once the
        # thinking channel and the post-EOS tail have been cut, exactly as for the final answer.
        if value is None and logits is not None:
            value = logits.argmax(-1)
        self.drafts.append(None if value is None else torch.as_tensor(value).detach().cpu())

    def put(self, value, **kwargs):
        pass

    def end(self, **kwargs):
        pass


def commit_analysis(argmax_steps, final_ids, maxlp_steps):
    """commit_step(p) = 1 + last step where argmax disagreed with final; conf = maxlp at commit."""
    A = torch.stack(argmax_steps)      # (S, B, C)
    LP = torch.stack(maxlp_steps)      # (S, B, C)
    S, B, C = A.shape
    n = min(C, final_ids.shape[-1])
    fin = final_ids[..., :n]           # (B, n)
    disagree = (A[:, :, :n] != fin.unsqueeze(0))         # (S,B,n)
    idx = torch.arange(S).view(S, 1, 1)
    last_bad = torch.where(disagree, idx, torch.full_like(A[:, :, :n], -1)).max(0).values  # (B,n)
    commit = (last_bad + 1).clamp(max=S - 1)
    conf = torch.gather(LP[:, :, :n], 0, commit.unsqueeze(0)).squeeze(0)  # (B,n)
    return commit, conf


# DiffusionGemma uses the Gemma 4 chat format: every reply opens with a thinking channel,
# "<|channel>thought\n...<channel|>", which is empty when thinking is disabled (the default;
# verified empty on every sampled TriviaQA/SamSum/WMT19 decode). The two markers are special
# tokens and vanish under skip_special_tokens, but "thought" and the newline are ordinary tokens,
# so a plain decode yields "thought\nFriday" and zeroes every exact-match metric. The channel is
# removed at the token level, so texts, token ids, likelihoods, per-position traces and drafts
# all describe the same answer tokens.
_CHANNEL_OPEN = "<|channel>"
_CHANNEL_CLOSE = "<channel|>"


def _answer_positions(ids: List[int], stop_ids, open_id: Optional[int], close_id: Optional[int]) -> List[int]:
    """Canvas positions of the answer: before the first EOS/pad, outside the thinking channel.

    Everything from ``open_id`` up to and including the next ``close_id`` is dropped; an
    unclosed channel (possible in intermediate drafts) is dropped up to the end. With
    ``open_id=None`` only the EOS/pad truncation is applied.
    """
    keep, inside = [], False
    for p, t in enumerate(ids):
        t = int(t)
        if t in stop_ids:
            break
        if inside:
            inside = t != close_id
            continue
        if open_id is not None and t == open_id:
            inside = True
            continue
        keep.append(p)
    return keep


def _dedup(seq: List[str]) -> List[str]:
    """Order-preserving de-duplication; drops ``None``/empty drafts (bounds NLI cost)."""
    out, seen = [], set()
    for s in seq:
        if s is None:
            continue
        s = str(s)
        if len(s) == 0 or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


class WhiteboxModelDiffusionGemma(Model):
    """Whitebox adapter for DiffusionGemma; duck-types ``WhiteboxModelDiffusionLLM``.

    ``model_type`` is hard-coded to ``"CausalLM"`` (load-bearing — the stat calculators
    key their slicing off it). The constructor ``tokenizer`` argument is the multimodal
    ``AutoProcessor``; the underlying text tokenizer is exposed as ``self.tokenizer`` so
    downstream code (``model.tokenizer.decode`` / ``.all_special_ids`` / ``.eos_token_id``)
    works unchanged, while the processor (chat template) is kept as ``self.processor``.
    """

    def __init__(
        self,
        model,
        tokenizer,
        generation_parameters: Optional[GenerationParameters] = None,
        instruct: bool = False,
        device: str = "cuda",
        length: int = 128,
        block_length: int = 32,
        eos_ids=(1, 106, 50),
        model_path: Optional[str] = None,
        strip_thought_prefix: bool = True,
        attention_device: Optional[str] = None,
        **kwargs,
    ):
        self.model = model
        # `tokenizer` is the AutoProcessor; keep the processor (chat template) and the
        # underlying text tokenizer separately.
        self.processor = tokenizer
        self.tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        if getattr(self.tokenizer, "pad_token", None) is None and getattr(
            self.tokenizer, "eos_token", None
        ) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.generation_parameters = generation_parameters or GenerationParameters()
        self.instruct = instruct
        # Drop the "<|channel>thought\n...<channel|>" thinking channel from answers, drafts
        # and all per-position stats. On by default because the leaked "thought\n" silently
        # zeroes any exact-match metric; see _answer_positions. Set False to reproduce runs
        # made before this was fixed.
        self.strip_thought_prefix = strip_thought_prefix
        self.base_device = device
        # Where attention maps are collected ("cpu" in all runs, as for LLaDA: long prompts OOM otherwise).
        self.attention_device = attention_device or device
        self.length = length
        self.block_length = block_length
        self.model_path = model_path

        # EOS ids: prefer the loaded checkpoint's own generation_config, else the
        # configured list ([1, 106, 50] for diffusiongemma-26B-A4B-it).
        gen_cfg = getattr(model, "generation_config", None)
        cfg_eos = getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
        if cfg_eos is None:
            self.eos_ids = list(eos_ids)
        elif isinstance(cfg_eos, int):
            self.eos_ids = [cfg_eos]
        else:
            self.eos_ids = list(cfg_eos)  # VERIFY@runtime: re-read from the loaded model
        self.eos_id_set = set(int(e) for e in self.eos_ids)

        # --- Load-bearing duck-type attributes (mirror WhiteboxModelDiffusionLLM) ---
        self.model_type = "CausalLM"        # hard-coded, load-bearing
        self.mask_id = None                 # DiffusionGemma has NO mask token
        self.use_cache = True               # informational only; NEVER passed to generate()
        self.shift_logits = False
        self.needs_attention_mask = False
        self.prompt_suffix = None
        self.steps = length
        # Stop strings from generation_params, truncated at token level as in WhiteboxModelDiffusionLLM.
        self.stop_sequences = list(getattr(self.generation_parameters, "stop_strings", None) or [])
        self.stop_token_ids = [
            self.tokenizer.encode(stop_seq, add_special_tokens=False)
            for stop_seq in self.stop_sequences
        ]

    # Attention / hidden-state formatting shared with the LLaDA backend.
    process_attention = WhiteboxModelDiffusionLLM.process_attention
    rearrange_hidden_states = WhiteboxModelDiffusionLLM.rearrange_hidden_states

    # ------------------------------------------------------------------ basics
    def device(self):
        return self.base_device

    def __call__(self, input_ids=None, attention_mask=None, output_attentions=False, output_hidden_states=False, **kwargs):
        """LM-style forward over a full token sequence, as calculators call the LLaDA model.

        DiffusionGemma's encoder is a causal Gemma 4 stack, so its pass over [prompt + answer]
        gives the autoregressive attention maps / hidden states that the attention calculators
        (AttentionForwardPass, AttentionElicitingPrompt) expect.
        """
        with _eager_attention(self.model, output_attentions), torch.inference_mode():
            return self.model.model.encoder(
                input_ids=input_ids, attention_mask=attention_mask,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            )

    def tokenize(
        self, texts: Union[List[str], List[List[Dict[str, str]]]]
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a batch (left-padded for generation).

        Instruct mode renders each prompt through the processor chat template with
        ``add_generation_prompt=True``; non-instruct mode tokenizes the raw strings.
        """
        tok = self.tokenizer
        prev_side = getattr(tok, "padding_side", "right")
        tok.padding_side = "left"  # left-pad for diffusion generation
        try:
            if self.instruct:
                conversations = [
                    ([{"role": "user", "content": chat}] if isinstance(chat, str) else chat)
                    for chat in texts
                ]
                try:
                    enc = self.processor.apply_chat_template(
                        conversations,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        add_generation_prompt=True,
                        padding=True,
                    )  # VERIFY@runtime: batched chat-template tokenisation + left padding
                except Exception:
                    # Bare-tokenizer fallback: render to strings, then pad-tokenize.
                    rendered = []
                    for conv in conversations:
                        try:
                            rendered.append(
                                self.processor.apply_chat_template(
                                    conv, add_generation_prompt=True, tokenize=False
                                )
                            )
                        except Exception:
                            rendered.append(conv[-1]["content"] if conv else "")
                    enc = tok(rendered, padding=True, return_tensors="pt", add_special_tokens=False)
            else:
                enc = tok(texts, padding=True, return_tensors="pt", add_special_tokens=True)
        finally:
            tok.padding_side = prev_side

        enc = dict(enc)
        return {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}

    # --------------------------------------------------------------- generation
    def generate(self, input_ids=None, max_new_tokens=None, min_new_tokens: int = 1, **kwargs):
        """Run DiffusionGemma generation with the read-only trace hooks.

        Swallows the standard harness kwargs (``output_scores``, ``return_dict_in_generate``,
        ``output_attentions``, ``output_hidden_states``, ``num_return_sequences``,
        ``suppress_tokens``, ``do_sample`` …) — DiffusionGemma has its own sampler and does
        not accept most of them — and forwards ONLY ``input_ids`` / ``max_new_tokens`` /
        ``attention_mask`` plus the processor + streamer. ``use_cache`` is NEVER passed.
        """
        from transformers import LogitsProcessorList

        if input_ids is None:
            input_ids = kwargs.pop("input_ids", None)
        if input_ids is None:
            raise ValueError("WhiteboxModelDiffusionGemma.generate requires input_ids")
        if max_new_tokens is None:
            max_new_tokens = self.length
        attention_mask = kwargs.get("attention_mask", None)

        # Opt-in stochastic decoding. DiffusionGemma commits argmax(processed_logits) and has
        # no multinomial path, so without this every decode of a prompt is identical and the
        # sampling-based estimators collapse. See GumbelNoiseLogitsProcessor.
        sample_temperature = kwargs.pop("sample_temperature", None)

        proc = UQTraceProcessor()
        strm = UQTraceStreamer(self.tokenizer)
        processors = [proc]  # UQTraceProcessor FIRST -> it records raw logits
        if sample_temperature is not None:
            processors.append(GumbelNoiseLogitsProcessor(float(sample_temperature)))
        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            logits_processor=LogitsProcessorList(processors),
            streamer=strm,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        # NEVER pass use_cache: DiffusionGemma always keeps the KV cache on.  # VERIFY@runtime

        with torch.inference_mode():
            out = self.model.generate(**gen_kwargs)

        return self._assemble_output(
            input_ids, out, proc, strm, max_new_tokens, attention_mask,
            kwargs.get("output_attentions", False), kwargs.get("output_hidden_states", False),
        )

    def _final_forward(self, input_ids, attention_mask, canvas, self_conditioning_logits=None,
                       output_attentions=False, output_hidden_states=False):
        """One clean pass over the final canvas, as WhiteboxModelDiffusionLLM.post_processing does for LLaDA.

        The prompt goes through the encoder (KV cache) and the final canvas through the decoder,
        self-conditioned on the last denoising step's logits with that step's temperature applied,
        exactly as generate() feeds them. The pass reproduces the denoiser's final prediction
        (argmax == answer at every answer position checked). Returns canvas log-probs and the
        encoder/decoder outputs (attentions / hidden states when requested).
        """
        with _eager_attention(self.model, output_attentions), torch.inference_mode():
            enc = self.model.model.encoder(
                input_ids=input_ids, attention_mask=attention_mask,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            )
            dec = self.model(
                past_key_values=enc.past_key_values, decoder_input_ids=canvas,
                self_conditioning_logits=self_conditioning_logits,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            )
        return dec.logits.log_softmax(-1), enc, dec

    def _truncate(self, ids, keep, max_new_tokens):
        """Cap the answer at max_new_tokens and cut at the first stop sequence (as for LLaDA)."""
        keep = keep[:max_new_tokens]
        toks = ids[keep].tolist()
        stop = len(toks)
        for stop_ids in self.stop_token_ids:
            for j in range(1, len(toks) - len(stop_ids) + 1):
                if stop_ids and toks[j : j + len(stop_ids)] == stop_ids:
                    stop = min(stop, j)
                    break
        return keep[:stop]

    def _assemble_output(
        self, input_ids, out, proc: UQTraceProcessor, strm: UQTraceStreamer, max_new_tokens,
        attention_mask=None, output_attentions=False, output_hidden_states=False,
    ):
        """Turn the raw output + collected trace into the standard UQ output object."""
        canvas = out.sequences[:, input_ids.shape[1]:]
        inp = input_ids.detach().cpu().long()
        B, prompt_len = inp.shape

        # --- .sequences must be (B, prompt+gen). DiffusionGemma's decoder canvas is the
        # generated/answer region; detect whether out.sequences already carries the prompt
        # prefix, else prepend it to honour the (B, prompt+gen) contract.  # VERIFY@runtime
        raw = out.sequences
        if not torch.is_tensor(raw):
            raw = torch.as_tensor(raw)
        raw = raw.detach().cpu().long()
        if raw.shape[1] >= prompt_len and torch.equal(raw[:, :prompt_len], inp):
            sequences = raw
            answer_ids = raw[:, prompt_len:]
        else:
            sequences = torch.cat([inp, raw], dim=1)
            answer_ids = raw
        gen_len = int(answer_ids.shape[1])

        # Degenerate guard: no steps recorded (should not happen on the real model).
        if proc.nfe == 0 or len(proc.argmax) == 0:
            empty4d = torch.zeros((B, 1, 0, max(gen_len, 1)))
            return types.SimpleNamespace(
                sequences=sequences, scores=None,
                greedy_tokens=[[] for _ in range(B)],
                greedy_texts=["" for _ in range(B)],
                greedy_log_likelihoods=[[] for _ in range(B)],
                nfe=np.array([0] * B),
                remask_count=torch.zeros((B, sequences.shape[1]), dtype=torch.int32),
                first_remask_step=torch.zeros((B, gen_len), dtype=torch.int32),
                generation_trace=[[] for _ in range(B)],
                full_generation_trace=[[] for _ in range(B)],
                last_block_generation_trace=[[] for _ in range(B)],
                block_generation_traces=[[] for _ in range(B)],
                gen_trace_logits=empty4d, gen_trace_entropy=empty4d,
                gen_trace_argmax=empty4d.to(torch.int32), gen_trace_top2gap=empty4d,
                confidence_at_commit=torch.full((B, gen_len), float("-inf")),
                mask_fraction_at_commit=torch.full((B, gen_len), float("nan")),
                mask_id=None,
            )

        # --- Stack per-step reductions -> (S, B, C). ------------------------------------
        A_stack = torch.stack(proc.argmax)               # (S, B, C) int32
        LP_stack = torch.stack(proc.maxlp).float()       # (S, B, C) log P(argmax)
        ENT_stack = torch.stack(proc.entropy).float()    # (S, B, C)
        GAP_stack = torch.stack(proc.top2gap).float()    # (S, B, C)
        S, _, C = A_stack.shape
        # Answer canvas positions [0:L] map to generated tokens 0..L-1 (decoder canvas).
        L = min(gen_len, C)

        # SINGLE-CANVAS ONLY. DiffusionGemma refills the canvas when a generation needs more
        # than canvas_length tokens, and the hooks then fire again for the new canvas. This
        # code stacks every step into one (S, B, C) block and indexes positions 0..L-1, so a
        # second canvas would be silently folded onto the first: commit_analysis would compare
        # later-canvas argmaxes against first-canvas tokens, and the trace, tokens, texts and
        # likelihoods would describe only the opening canvas. Rather than emit corrupted
        # statistics, refuse. All reported runs used max_new_tokens=128 against canvas_length
        # 256 (longest observed generation 106 tokens), so they are single-canvas.
        if gen_len > C:
            raise RuntimeError(
                f"DiffusionGemma produced {gen_len} tokens but the trace covers a single "
                f"{C}-token canvas. Multi-canvas generation is not supported by this "
                f"backend: the per-canvas hook output cannot be folded into one trace "
                f"without corrupting commit steps and likelihoods. Reduce max_new_tokens to "
                f"at most {C}, or add per-canvas trace handling."
            )

        # --- Commit detection (argmax-stability) aligned to the ANSWER region. ----------
        # commit_analysis is reused verbatim; final_ids = answer_ids so canvas position p
        # aligns to generated token p.  # VERIFY@runtime (if the canvas embeds the prompt
        # prefix instead, pass `sequences` here and slice [:, prompt_len:] afterwards).
        commit, conf = commit_analysis(proc.argmax, answer_ids, proc.maxlp)  # (B, L), (B, L)

        # --- Answer positions: before the first EOS/pad, without the thinking channel. ----
        # Every per-position stat below is built on these canvas positions and left-aligned,
        # so index k of greedy_tokens, greedy_log_likelihoods, confidence_at_commit and the
        # gen_trace_* tensors refers to the same token (the trace estimators select positions
        # 0..len(greedy_tokens)-1).
        stop_ids = self.eos_id_set | {0}
        open_id = close_id = None
        if self.strip_thought_prefix:
            open_id = self.tokenizer.convert_tokens_to_ids(_CHANNEL_OPEN)
            close_id = self.tokenizer.convert_tokens_to_ids(_CHANNEL_CLOSE)
        # An empty answer keeps its end-of-turn token (as LLaDA keeps the EOS), and an answer that never
        # leaves the thinking channel keeps canvas position 0, so scores are never empty.
        keeps = [
            self._truncate(
                answer_ids[i],
                _answer_positions(answer_ids[i, :L].tolist(), stop_ids, open_id, close_id),
                max_new_tokens,
            )
            or _answer_positions(answer_ids[i, :L].tolist(), set(), open_id, close_id)[:1]
            or [0]
            for i in range(B)
        ]

        # --- Pack the four per-step traces to (B, num_canvases, max_steps, canvas_length).
        # Single canvas (num_canvases=1, max_steps=S); positions outside the answer are NaN
        # (-1 for argmax).
        def _pack(stack, fill):  # (S, B, C) -> (B, 1, S, L)
            out = torch.full((B, 1, S, L), fill, dtype=stack.dtype)
            for i, keep in enumerate(keeps):
                out[i, 0, :, : len(keep)] = stack[:, i, keep]
            return out

        gen_trace_logits = _pack(LP_stack, float("nan"))
        gen_trace_entropy = _pack(ENT_stack, float("nan"))
        gen_trace_top2gap = _pack(GAP_stack, float("nan"))
        gen_trace_argmax = _pack(A_stack, -1).to(torch.int32)

        # --- Per-answer-position metrics. -------------------------------------------------
        confidence_at_commit = torch.full((B, gen_len), float("-inf"))
        mask_fraction_at_commit = torch.full((B, gen_len), float("nan"))
        first_remask_step = torch.zeros((B, gen_len), dtype=torch.int32)
        greedy_tokens: List[List[int]] = []
        greedy_texts: List[str] = []
        greedy_lls: List[List[float]] = []

        for i, keep in enumerate(keeps):
            n = len(keep)
            ids = answer_ids[i, keep].tolist()
            greedy_tokens.append(ids)
            greedy_texts.append(self.tokenizer.decode(ids, skip_special_tokens=True))

            # Decision 1: greedy_log_likelihoods == committed-token log-prob per position.
            conf_i = conf[i, keep].float()
            confidence_at_commit[i, :n] = conf_i
            greedy_lls.append(conf_i.tolist())

            # renoise-fraction analog (~ mask_fraction_at_commit) + first-flip step.
            fin = answer_ids[i, keep]                            # (n,)
            Ai = A_stack[:, i, keep]                             # (S, n)
            noisy = (Ai != fin.unsqueeze(0)).float().mean(dim=1)             # (S,)
            commit_i = commit[i, keep].long().clamp(0, S - 1)               # (n,)
            mask_fraction_at_commit[i, :n] = noisy[commit_i]
            if S > 1:
                changed = Ai[1:] != Ai[:-1]                      # (S-1, n)
                has = changed.any(dim=0)
                fr = torch.where(
                    has,
                    changed.float().argmax(dim=0).to(torch.int32) + 1,
                    torch.zeros(n, dtype=torch.int32),
                )
                first_remask_step[i, :n] = fr

        # --- remask_count (B, prompt+gen): #argmax flips per canvas position. ------------
        ncol = min(C, sequences.shape[1])
        if S > 1:
            remask_count = (A_stack[1:, :, :ncol] != A_stack[:-1, :, :ncol]).sum(0).to(torch.int32)
        else:
            remask_count = torch.zeros((B, ncol), dtype=torch.int32)

        # --- Deduped draft traces (List[batch][List[str]]). -----------------------------
        # Single-canvas: generation/full/last_block collapse to the deduped per-step
        # drafts; block_generation_traces has one block. Each draft is cut like the final
        # answer (thinking channel removed, truncated at the first EOS/pad).
        drafts = strm.drafts
        generation_trace, full_generation_trace = [], []
        last_block_generation_trace, block_generation_traces = [], []
        for i in range(B):
            per_step = []
            for d in drafts:
                if d is None or i >= d.shape[0]:
                    continue
                keep = self._truncate(
                    d[i], _answer_positions(d[i].tolist(), stop_ids, open_id, close_id), max_new_tokens
                )
                per_step.append(self.tokenizer.decode(d[i, keep].tolist(), skip_special_tokens=True))
            # An empty answer leaves no non-empty drafts; keep the final answer so the trace is never
            # empty (trace calculators fail on an empty list and the harness then drops the estimator).
            ded = _dedup(per_step) or [greedy_texts[i]]
            full_generation_trace.append(ded)
            generation_trace.append(ded)
            last_block_generation_trace.append(ded)
            block_generation_traces.append([ded] if ded else [])

        nfe = np.array([int(proc.nfe)] * B)

        # --- LLaDA-format output (WhiteboxModelDiffusionLLM.post_processing): prompt + answer
        # sequences, per-token log-prob tuple, and attentions / hidden states from one clean pass
        # over the final canvas, so DiffusionGreedyProbsCalculator and the sampling, attention,
        # embedding and training-statistics calculators run unchanged.
        cfg = self.model.generation_config  # temperature of the last step, as generate() applies it
        t = cfg.t_min + (cfg.t_max - cfg.t_min) * float(proc.last_step) / cfg.max_denoising_steps
        n_max = max(len(keep) for keep in keeps)
        # As in WhiteboxModelDiffusionLLM.post_processing, the clean pass sees at most 2048 tokens:
        # the prompt is truncated from the left, which bounds eager-attention memory on long inputs (XSum).
        fwd_ids = input_ids[:, -max(1, 2048 - n_max):]
        fwd_mask = None if attention_mask is None else attention_mask[:, -fwd_ids.shape[1]:]
        fwd_prompt_len = fwd_ids.shape[1]
        logp, enc, dec = self._final_forward(
            fwd_ids, fwd_mask, canvas, (proc.last_scores / t).to(self.model.dtype),
            output_attentions, output_hidden_states,
        )
        eos = self.tokenizer.eos_token_id
        scores = torch.zeros((B, n_max, logp.shape[-1]), dtype=torch.float16)
        answers = torch.full((B, n_max), eos, dtype=torch.long)
        for i, keep in enumerate(keeps):
            scores[i, : len(keep)] = logp[i, keep].to(torch.float16).cpu()
            answers[i, : len(keep)] = answer_ids[i, keep]
        scores = tuple(scores.transpose(0, 1))
        sequences = torch.cat([inp, answers], dim=1)

        # Attentions / hidden states over the virtual sequence [prompt, answer] (batch_size=1,
        # as in all runs): prompt rows from the encoder, answer rows from the decoder, whose
        # attention covers the prompt KV cache followed by the canvas.
        attentions, hidden_states, n = [], None, len(keeps[0])
        if (output_attentions or output_hidden_states) and B != 1:
            raise NotImplementedError("attentions / hidden states need batch_size=1")
        if output_attentions:
            keep = torch.tensor(keeps[0], device=canvas.device)
            full = []
            for ea, da in zip(enc.attentions, dec.attentions):
                # Sliding-window layers cache only the last `kv` prompt keys (window 1024).
                kv = da.shape[-1] - canvas.shape[1]
                rows = da.new_zeros(*da.shape[:2], n, fwd_prompt_len + n)
                rows[..., fwd_prompt_len - kv : fwd_prompt_len] = da[:, :, keep, :kv]
                rows[..., fwd_prompt_len:] = da[:, :, keep][..., kv + keep]
                full.append(torch.cat([torch.nn.functional.pad(ea, (0, n)), rows], dim=2))
            attentions = [self.process_attention(full, t, fwd_prompt_len) for t in range(n)]
        if output_hidden_states:
            hs = tuple(
                torch.cat([eh, dh[:, keeps[0]]], dim=1)
                for eh, dh in zip(enc.hidden_states, dec.hidden_states)
            )
            hidden_states = self.rearrange_hidden_states(hs, fwd_prompt_len, n)
        del logp, enc, dec

        output = GenerateDecoderOnlyOutput(
            sequences=sequences,
            logits=scores,
            scores=scores,
            attentions=attentions,
            hidden_states=hidden_states,
        )
        for name, value in dict(
            greedy_tokens=greedy_tokens,
            greedy_texts=greedy_texts,
            greedy_log_likelihoods=greedy_lls,
            nfe=nfe,
            remask_count=remask_count,
            first_remask_step=first_remask_step,
            generation_trace=generation_trace,
            full_generation_trace=full_generation_trace,
            last_block_generation_trace=last_block_generation_trace,
            block_generation_traces=block_generation_traces,
            gen_trace_logits=gen_trace_logits,
            gen_trace_entropy=gen_trace_entropy,
            gen_trace_argmax=gen_trace_argmax,
            gen_trace_top2gap=gen_trace_top2gap,
            confidence_at_commit=confidence_at_commit,
            mask_fraction_at_commit=mask_fraction_at_commit,
            mask_id=None,  # DiffusionGemma has no mask token
        ).items():
            setattr(output, name, value)

        del A_stack, LP_stack, ENT_stack, GAP_stack
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return output

    # ------------------------------------------------------- text-level generation
    def _legacy_generate_texts(self, *args, **kwargs):
        outputs = self.generate(*args, **kwargs)
        return [
            self.tokenizer.decode(seq, skip_special_tokens=True)
            for seq in outputs.sequences
        ]

    def _is_text_generation_input(self, input_texts):
        if isinstance(input_texts, str):
            return True
        if not isinstance(input_texts, (list, tuple)):
            return False
        return all(
            isinstance(text, str)
            or (isinstance(text, list) and all(isinstance(m, dict) for m in text))
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

        input_texts = [input_texts] if isinstance(input_texts, str) else list(input_texts)

        num_return_sequences = int(generation_kwargs.pop("num_return_sequences", 1) or 1)
        if "min_new_tokens" not in generation_kwargs:
            generation_kwargs["min_new_tokens"] = generation_kwargs.pop("min_length", 1)
        else:
            generation_kwargs.pop("min_length", None)

        if num_return_sequences > 1:
            input_texts = [t for t in input_texts for _ in range(num_return_sequences)]

        batch = self.tokenize(input_texts)
        batch = {k: v.to(self.device()) for k, v in batch.items()}
        out = self.generate(**batch, **generation_kwargs)
        sequences = out.sequences.cpu()
        input_len = batch["input_ids"].shape[1]
        return [
            self.tokenizer.decode(seq[input_len:], skip_special_tokens=True)
            for seq in sequences
        ]
