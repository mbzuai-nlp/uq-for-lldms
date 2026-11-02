"""End-to-end validation through WhiteboxModelDiffusionLLM adapter.

Confirms that the production code path the polygraph_eval entrypoint uses
produces correct outputs after switching the cache strategy for Dream.

Run:
    CUDA_VISIBLE_DEVICES=0 python scripts/validate_dream_adapter.py
"""

from __future__ import annotations

import os
import sys
import time

import torch
from lm_polygraph.utils.generation_parameters import GenerationParameters

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from llada.load_dream_cached import load_model, load_tokenizer  # noqa: E402
from whitebox_model_dllm import WhiteboxModelDiffusionLLM  # noqa: E402

MODEL_PATH = "Dream-org/Dream-v0-Instruct-7B"
MASK_ID = 151666

PROMPTS = [
    "What is the capital of France?",
    "What is 17 multiplied by 23?",
    "Name the largest planet in our solar system.",
    "Summarize in one sentence: The sun rose over the quiet meadow as birds began their morning chorus, filling the air with cheerful melodies.",
    "List three primary colors.",
]


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(MODEL_PATH, add_bos_token=False)
    model = load_model(MODEL_PATH, device_map=device)

    gp = GenerationParameters(temperature=0.0)

    # Mirror polygraph defaults that drive the Dream cached eval.
    common_kwargs = dict(
        steps=128,
        length=128,
        block_length=32,
        attention_device="cuda",
        mask_id=MASK_ID,
        prompt_suffix=None,
        shift_logits=True,
    )

    cached = WhiteboxModelDiffusionLLM(
        model=model,
        tokenizer=tokenizer,
        generation_parameters=gp,
        instruct=True,
        device=device,
        use_cache=True,
        **common_kwargs,
    )
    no_cache = WhiteboxModelDiffusionLLM(
        model=model,
        tokenizer=tokenizer,
        generation_parameters=gp,
        instruct=True,
        device=device,
        use_cache=False,
        **common_kwargs,
    )

    failures = 0
    for i, prompt in enumerate(PROMPTS):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
        ).to(device)

        kwargs = dict(
            input_ids=input_ids,
            min_new_tokens=1,
            max_new_tokens=128,
            do_sample=False,
            output_attentions=False,
            output_hidden_states=False,
        )
        t0 = time.time()
        out_c = cached.generate(**kwargs)
        t_c = time.time() - t0
        t0 = time.time()
        out_nc = no_cache.generate(**kwargs)
        t_nc = time.time() - t0

        # WhiteboxModelDiffusionLLM.post_processing returns a GenerateDecoderOnlyOutput.
        seq_c = out_c.sequences if hasattr(out_c, "sequences") else out_c
        seq_nc = out_nc.sequences if hasattr(out_nc, "sequences") else out_nc

        prompt_len = int(input_ids.shape[1])
        text_c = tokenizer.decode(seq_c[0, prompt_len:], skip_special_tokens=True)
        text_nc = tokenizer.decode(seq_nc[0, prompt_len:], skip_special_tokens=True)
        for stop in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
            text_c = text_c.split(stop, 1)[0]
            text_nc = text_nc.split(stop, 1)[0]
        text_c = text_c.strip()
        text_nc = text_nc.strip()

        # Hard regression check: cache output must not be a strict prefix of
        # no-cache output more than 50% shorter (catches "The sun" vs full).
        bad = False
        why = ""
        if not text_c:
            bad, why = True, "cache empty"
        elif "<|mask|>" in text_c:
            bad, why = True, "cache has <|mask|>"
        elif (
            text_nc
            and len(text_c) < 0.5 * len(text_nc)
            and text_nc.startswith(text_c)
        ):
            bad, why = True, "cache prefix-truncates no-cache by >50%"

        if bad:
            failures += 1

        print(f"\n=== Prompt {i + 1}/{len(PROMPTS)} ===")
        print(f"Q: {prompt}")
        print(f"  no-cache ({t_nc:5.2f}s): {text_nc!r}")
        print(f"  cache    ({t_c:5.2f}s): {text_c!r}")
        if bad:
            print(f"  !! REGRESSION: {why}")

    print()
    print(f"Adapter validation: {len(PROMPTS) - failures}/{len(PROMPTS)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
