"""Quick validation: Dream 7B with-cache vs no-cache generation parity.

Loads Dream-org/Dream-v0-Instruct-7B via the cache-aware Fast-dLLM model code,
runs a handful of fixed prompts twice (cache=ON via generate_with_dual_cache,
cache=OFF via generate), and reports any obviously-broken outputs.

Run:
    CUDA_VISIBLE_DEVICES=0 python scripts/validate_dream_cache.py
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import List, Tuple

import torch

# Make repo root importable so `llada.*` and `scripts/*` can be loaded as modules.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from llada.generate import (  # noqa: E402
    generate,
    generate_with_dual_cache,
    generate_with_prefix_cache,
)
from llada.load_dream_cached import load_model, load_tokenizer  # noqa: E402

# Selects which cached path to validate against the no-cache baseline.
# Values: "prefix" (default, structurally correct) or "dual".
CACHE_PATH = os.environ.get("DREAM_CACHE_PATH", "prefix")


MODEL_PATH = "Dream-org/Dream-v0-Instruct-7B"
MASK_TOKEN = "<|mask|>"
MASK_ID = 151666

# Generation defaults — match production whitebox adapter settings.
GEN_LENGTH = int(os.environ.get("DREAM_GEN_LENGTH", "128"))
STEPS = int(os.environ.get("DREAM_STEPS", "128"))
BLOCK_LENGTH = int(os.environ.get("DREAM_BLOCK_LENGTH", "32"))
TEMPERATURE = 0.0  # deterministic
SHIFT_LOGITS = True  # Dream uses AR-shifted predictions

_COQA_STORY = (
    "Once upon a time, in a barn near a farm house, there lived a little white "
    "kitten named Cotton. Cotton lived high up in a nice warm place above the "
    "barn where all of the farmer's horses slept. But Cotton wasn't alone in "
    "her little home above the barn, oh no. She shared her hay bed with her "
    "mommy and 5 other sisters. All of her sisters were cute and fluffy, like "
    "Cotton. But she was the only white one in the bunch. The rest of her "
    "sisters were all orange with beautiful white tiger stripes like Cotton's "
    "mommy."
)

PROMPTS: List[str] = [
    "What is the capital of France?",
    "Who wrote the play 'Romeo and Juliet'?",
    "What is 17 multiplied by 23?",
    "Name the largest planet in our solar system.",
    "In one short sentence, what does photosynthesis do?",
    "List three primary colors.",
    "What is the chemical symbol for gold?",
    "How many continents are there on Earth?",
    "Who painted the Mona Lisa?",
    "What gas do plants absorb from the atmosphere?",
    "Convert 100 degrees Fahrenheit to Celsius (just the number).",
    "What year did World War II end?",
    "Name the smallest prime number greater than 10.",
    "What is the speed of light in vacuum (km/s, approximate)?",
    f"{_COQA_STORY}\n\nQuestion: What color was Cotton?",
    f"{_COQA_STORY}\n\nQuestion: Where did Cotton live?",
    f"{_COQA_STORY}\n\nQuestion: How many sisters did Cotton have?",
    f"{_COQA_STORY}\n\nQuestion: What color were Cotton's sisters?",
    "Translate to English: 'Le chat est sur la table.'",
    "Summarize in one sentence: The sun rose over the quiet meadow as birds began their morning chorus, filling the air with cheerful melodies.",
]


def is_garbage(text: str) -> Tuple[bool, str]:
    """Heuristic check. Returns (is_garbage, reason)."""
    stripped = text.strip()
    if not stripped:
        return True, "empty"
    if MASK_TOKEN in text:
        return True, f"contains {MASK_TOKEN}"
    # 4-gram repeat ≥3 consecutive times
    tokens = stripped.split()
    if len(tokens) >= 12:
        for n in (1, 2, 3, 4):
            for i in range(len(tokens) - 3 * n):
                a = tokens[i : i + n]
                b = tokens[i + n : i + 2 * n]
                c = tokens[i + 2 * n : i + 3 * n]
                if a == b == c:
                    return True, f"repeats {' '.join(a)!r} x3"
    # Excessive repeated punctuation/chars (e.g. ".......")
    if re.search(r"(.)\1{15,}", stripped):
        return True, "char repeats >15"
    return False, ""


def build_prompt_ids(tokenizer, user_msg: str, device: str) -> torch.Tensor:
    messages = [{"role": "user", "content": user_msg}]
    ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(device)
    return ids


def decode_completion(
    tokenizer, output_ids: torch.Tensor, prompt_len: int
) -> str:
    completion = output_ids[0, prompt_len:]
    text = tokenizer.decode(completion, skip_special_tokens=False)
    # Strip stop sequences (ChatML)
    for stop in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    # Re-decode with skip_special_tokens for the visible string
    pretty = tokenizer.decode(completion, skip_special_tokens=True)
    for stop in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        idx = pretty.find(stop)
        if idx != -1:
            pretty = pretty[:idx]
    return pretty.strip()


def run_one(model, tokenizer, prompt_ids, use_cache: bool):
    block_length = min(BLOCK_LENGTH, GEN_LENGTH)
    common = dict(
        steps=STEPS,
        gen_length=GEN_LENGTH,
        block_length=block_length,
        temperature=TEMPERATURE,
        remasking="low_confidence",
        threshold=0.9,
        factor=1,
        mask_id=MASK_ID,
        shift_logits=SHIFT_LOGITS,
    )
    t0 = time.time()
    with torch.inference_mode():
        if use_cache:
            if CACHE_PATH == "dual":
                out = generate_with_dual_cache(
                    model, tokenizer, prompt_ids, **common
                )
            else:
                out = generate_with_prefix_cache(
                    model, prompt_ids, tokenizer=tokenizer, **common
                )
        else:
            out = generate(
                model,
                prompt_ids,
                tokenizer=tokenizer,
                needs_attention_mask=False,
                **common,
            )
    elapsed = time.time() - t0
    # `generate` returns a tuple (sequence, *trace_stats); first element is ids.
    if isinstance(out, tuple):
        seq = out[0]
    else:
        seq = out
    return seq, elapsed


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] device={device} model={MODEL_PATH} cache_path={CACHE_PATH}")
    tokenizer = load_tokenizer(MODEL_PATH, add_bos_token=False)
    model = load_model(MODEL_PATH, device_map=device)
    print(f"[load] mask_token_id={tokenizer.mask_token_id} (using {MASK_ID})")
    if tokenizer.mask_token_id != MASK_ID:
        print(
            f"[warn] tokenizer.mask_token_id={tokenizer.mask_token_id} "
            f"differs from hardcoded {MASK_ID}"
        )

    rows = []
    for i, prompt in enumerate(PROMPTS):
        prompt_ids = build_prompt_ids(tokenizer, prompt, device)
        prompt_len = int(prompt_ids.shape[1])

        seq_nc, t_nc = run_one(model, tokenizer, prompt_ids, use_cache=False)
        text_nc = decode_completion(tokenizer, seq_nc, prompt_len)
        bad_nc, why_nc = is_garbage(text_nc)

        seq_c, t_c = run_one(model, tokenizer, prompt_ids, use_cache=True)
        text_c = decode_completion(tokenizer, seq_c, prompt_len)
        bad_c, why_c = is_garbage(text_c)

        match = text_nc == text_c
        rows.append(
            {
                "i": i,
                "prompt": prompt,
                "no_cache": text_nc,
                "cache": text_c,
                "bad_nc": bad_nc,
                "why_nc": why_nc,
                "bad_c": bad_c,
                "why_c": why_c,
                "match": match,
                "t_nc": t_nc,
                "t_c": t_c,
            }
        )

        print()
        print(f"=== Prompt {i + 1}/{len(PROMPTS)} ===")
        print(f"Q: {prompt}")
        print(f"  no-cache ({t_nc:5.2f}s){' [GARBAGE: ' + why_nc + ']' if bad_nc else ''}: {text_nc!r}")
        print(f"  cache    ({t_c:5.2f}s){' [GARBAGE: ' + why_c + ']' if bad_c else ''}: {text_c!r}")
        print(f"  exact-match: {match}")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n = len(rows)
    nc_bad = sum(r["bad_nc"] for r in rows)
    c_bad = sum(r["bad_c"] for r in rows)
    matches = sum(r["match"] for r in rows)
    avg_t_nc = sum(r["t_nc"] for r in rows) / n
    avg_t_c = sum(r["t_c"] for r in rows) / n
    print(f"  prompts:                {n}")
    print(f"  garbage no-cache:       {nc_bad}/{n}")
    print(f"  garbage cache:          {c_bad}/{n}")
    print(f"  exact-match cache==no:  {matches}/{n}")
    print(f"  avg latency no-cache:   {avg_t_nc:.2f}s")
    print(f"  avg latency cache:      {avg_t_c:.2f}s")
    if avg_t_nc > 0:
        print(f"  cache speedup:          {avg_t_nc / max(avg_t_c, 1e-6):.2f}x")
    # Exit non-zero only if cache produces garbage no-cache did not.
    cache_only_garbage = sum(
        1 for r in rows if r["bad_c"] and not r["bad_nc"]
    )
    print(f"  cache-only garbage:     {cache_only_garbage}/{n}")
    return 0 if cache_only_garbage == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
