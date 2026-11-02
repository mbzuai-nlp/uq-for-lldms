"""Multi-dataset Dream cache vs no-cache validation through production adapter.

For each of the 9 supported datasets we:
  1. Stream `subsample` prompts from `LM-Polygraph/<dataset>` test split.
  2. Format them per-dataset (chat template for instruct, raw for WMT minimal).
  3. Generate via WhiteboxModelDiffusionLLM both with use_cache=True (the new
     Dream prefix-cache path after the fix) and with use_cache=False.
  4. Compare per-prompt: garbage heuristics + exact-match + length-ratio.
  5. Emit a per-dataset markdown row and a final summary, ready for a PR comment.

Run:
    CUDA_VISIBLE_DEVICES=0 python scripts/validate_dream_datasets.py \
        [--subsample 20] [--datasets coqa triviaqa ...]
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from typing import List, Optional, Tuple

import torch
from datasets import load_dataset
from lm_polygraph.utils.generation_parameters import GenerationParameters

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from llada.load_dream_cached import load_model, load_tokenizer  # noqa: E402
from whitebox_model_dllm import WhiteboxModelDiffusionLLM  # noqa: E402


MODEL_PATH = "Dream-org/Dream-v0-Instruct-7B"
MASK_ID = 151666
MASK_TOKEN = "<|mask|>"


@dataclasses.dataclass
class DatasetSpec:
    name: str
    repo: str
    subset: str
    instruct: bool
    max_new_tokens: int
    # Stop sequences (text-level) appended after model stop-strings.
    stop_strings: Tuple[str, ...] = ()
    # Per-dataset input rewrite to match production yaml semantics.
    # For WMT minimal configs polygraph_eval extracts the source from
    # "Original:\n{text}\nTranslation:" and rewrites to "{lang}: {text} = English:"
    # (see polygraph_eval lines 173-183).
    minimal_translation_lang: Optional[str] = None


# Production settings extracted from examples/configs/polygraph_eval_*.yaml
DATASETS: List[DatasetSpec] = [
    DatasetSpec("coqa",        "LM-Polygraph/coqa",        "simple_instruct",        True,  64,  ("\n",)),
    DatasetSpec("triviaqa",    "LM-Polygraph/triviaqa",    "simple_instruct",        True,  64,  ()),
    DatasetSpec("mmlu",        "LM-Polygraph/mmlu",        "simple_instruct",        True,  32,  ("\n",)),
    DatasetSpec("gsm8k",       "LM-Polygraph/gsm8k",       "simple_instruct",        True,  256, ()),
    DatasetSpec("samsum",      "LM-Polygraph/samsum",      "simple_instruct",        True,  128, ()),
    DatasetSpec("truthfulqa",  "LM-Polygraph/truthfulqa",  "simple_instruct",        True,  64,  ()),
    DatasetSpec("xsum",        "LM-Polygraph/xsum",        "simple_instruct",        True,  64,  ()),
    DatasetSpec("wmt19_deen",  "LM-Polygraph/wmt19",       "deen_simple_instruct",   False, 128,
                ("\nGerman:", "\n\nGerman:", "\n\n"),
                minimal_translation_lang="German"),
    DatasetSpec("wmt14_fren",  "LM-Polygraph/wmt14",       "fren_simple_instruct",   False, 128,
                ("\nFrench:", "\n\nFrench:", "\n\n"),
                minimal_translation_lang="French"),
]


_TRANSLATION_RE = None


def maybe_rewrite_minimal(text: str, lang: Optional[str]) -> str:
    """Apply the production minimal-translation rewrite when configured."""
    global _TRANSLATION_RE
    if not lang:
        return text
    import re
    if _TRANSLATION_RE is None:
        _TRANSLATION_RE = re.compile(r"Original:\n(.*)\nTranslation:", re.DOTALL)
    m = _TRANSLATION_RE.search(text)
    if m:
        return f"{lang}: {m.group(1)} = English:"
    return text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", type=int, default=20)
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Subset of dataset names to run (default: all 9).")
    p.add_argument("--out", default=None, help="Markdown report output path.")
    return p.parse_args()


def is_garbage(text_c: str, text_nc: str) -> Tuple[bool, str]:
    """Detect cache-only regressions. Returns (bad, reason)."""
    stripped = text_c.strip()
    if not stripped and text_nc.strip():
        return True, "cache empty"
    if MASK_TOKEN in text_c:
        return True, f"cache contains {MASK_TOKEN}"
    if (
        text_nc.strip()
        and 0 < len(stripped) < 0.5 * len(text_nc.strip())
        and text_nc.strip().startswith(stripped)
    ):
        return True, "cache prefix-truncates no-cache >50%"
    return False, ""


def build_input_ids(tokenizer, prompt: str, instruct: bool, device: str) -> torch.Tensor:
    if instruct:
        msgs = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True
        )
    else:
        # Minimal translation prompt: dataset already provides "German: X = English:"
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    return ids.to(device)


def decode(tokenizer, seq: torch.Tensor, prompt_len: int, stops: Tuple[str, ...]) -> str:
    text = tokenizer.decode(seq[0, prompt_len:], skip_special_tokens=True)
    for stop in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        text = text.split(stop, 1)[0]
    for stop in stops:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def sample_prompts(spec: DatasetSpec, n: int) -> List[Tuple[str, str]]:
    """Stream first n samples; return list of (input, gold) with prompt rewrite."""
    ds = load_dataset(spec.repo, spec.subset, split="test", streaming=True, trust_remote_code=True)
    out: List[Tuple[str, str]] = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        text = maybe_rewrite_minimal(str(row["input"]), spec.minimal_translation_lang)
        out.append((text, str(row["output"])))
    return out


def run_dataset(
    spec: DatasetSpec,
    cached: WhiteboxModelDiffusionLLM,
    no_cache: WhiteboxModelDiffusionLLM,
    tokenizer,
    device: str,
    subsample: int,
) -> dict:
    print(f"\n>>> {spec.name}  ({spec.repo}/{spec.subset})  max_new_tokens={spec.max_new_tokens}")
    prompts = sample_prompts(spec, subsample)

    rows = []
    t_c_total = 0.0
    t_nc_total = 0.0
    for i, (prompt, gold) in enumerate(prompts):
        input_ids = build_input_ids(tokenizer, prompt, spec.instruct, device)
        prompt_len = int(input_ids.shape[1])
        kw = dict(
            input_ids=input_ids,
            min_new_tokens=1,
            max_new_tokens=spec.max_new_tokens,
            do_sample=False,
            output_attentions=False,
            output_hidden_states=False,
        )
        t0 = time.time()
        out_c = cached.generate(**kw)
        t_c = time.time() - t0
        t0 = time.time()
        out_nc = no_cache.generate(**kw)
        t_nc = time.time() - t0
        t_c_total += t_c
        t_nc_total += t_nc

        seq_c = out_c.sequences if hasattr(out_c, "sequences") else out_c
        seq_nc = out_nc.sequences if hasattr(out_nc, "sequences") else out_nc
        text_c = decode(tokenizer, seq_c, prompt_len, spec.stop_strings)
        text_nc = decode(tokenizer, seq_nc, prompt_len, spec.stop_strings)
        bad, why = is_garbage(text_c, text_nc)
        match = text_c == text_nc
        rows.append({
            "i": i, "prompt": prompt, "gold": gold,
            "no_cache": text_nc, "cache": text_c,
            "match": match, "bad": bad, "why": why,
            "len_nc": len(text_nc), "len_c": len(text_c),
        })
        flag = " !!" if bad else ""
        print(f"  [{i:2d}]{flag} match={match}  c({t_c:4.2f}s)={text_c[:80]!r:.80}  nc({t_nc:4.2f}s)={text_nc[:80]!r:.80}")

    n = len(rows)
    n_match = sum(r["match"] for r in rows)
    n_bad = sum(r["bad"] for r in rows)
    avg_len_ratio = (
        sum((r["len_c"] / max(r["len_nc"], 1)) for r in rows) / max(n, 1)
    )
    return {
        "spec": spec,
        "n": n,
        "exact_match": n_match,
        "garbage": n_bad,
        "avg_len_ratio": avg_len_ratio,
        "avg_t_c": t_c_total / max(n, 1),
        "avg_t_nc": t_nc_total / max(n, 1),
        "rows": rows,
    }


def main() -> int:
    args = parse_args()

    selected = (
        DATASETS if args.datasets is None
        else [d for d in DATASETS if d.name in args.datasets]
    )
    if not selected:
        print(f"No matching datasets in {[d.name for d in DATASETS]}")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(MODEL_PATH, add_bos_token=False)
    model = load_model(MODEL_PATH, device_map=device)

    gp = GenerationParameters(temperature=0.0)
    common = dict(
        steps=128,
        length=128,
        block_length=32,
        attention_device="cuda",
        mask_id=MASK_ID,
        prompt_suffix=None,
        shift_logits=True,
    )
    cached = WhiteboxModelDiffusionLLM(
        model=model, tokenizer=tokenizer, generation_parameters=gp,
        instruct=True, device=device, use_cache=True, **common,
    )
    no_cache = WhiteboxModelDiffusionLLM(
        model=model, tokenizer=tokenizer, generation_parameters=gp,
        instruct=True, device=device, use_cache=False, **common,
    )

    results = []
    for spec in selected:
        results.append(run_dataset(spec, cached, no_cache, tokenizer, device, args.subsample))

    # Markdown report
    report_lines = []
    report_lines.append("## Dream cache validation across datasets\n")
    report_lines.append(
        f"Subsample: {args.subsample} prompts per dataset, "
        f"`shift_logits=True`, `use_cache=True` routes through `generate_with_prefix_cache` "
        "(LLaDA path unchanged).\n"
    )
    report_lines.append(
        "| Dataset | n | exact-match | cache regressions | avg len-ratio (c/nc) | avg t cache | avg t no-cache | speedup |"
    )
    report_lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    total_n = total_match = total_bad = 0
    sum_ratio = sum_tc = sum_tnc = 0.0
    for r in results:
        s = r["spec"]
        total_n += r["n"]
        total_match += r["exact_match"]
        total_bad += r["garbage"]
        sum_ratio += r["avg_len_ratio"] * r["n"]
        sum_tc += r["avg_t_c"] * r["n"]
        sum_tnc += r["avg_t_nc"] * r["n"]
        speedup = (r["avg_t_nc"] / r["avg_t_c"]) if r["avg_t_c"] > 0 else float("nan")
        report_lines.append(
            f"| {s.name} | {r['n']} | {r['exact_match']}/{r['n']} "
            f"| {r['garbage']}/{r['n']} "
            f"| {r['avg_len_ratio']:.2f} "
            f"| {r['avg_t_c']:.2f}s "
            f"| {r['avg_t_nc']:.2f}s "
            f"| {speedup:.2f}x |"
        )
    overall_ratio = sum_ratio / max(total_n, 1)
    overall_speedup = (sum_tnc / sum_tc) if sum_tc > 0 else float("nan")
    report_lines.append(
        f"| **TOTAL** | **{total_n}** | **{total_match}/{total_n}** "
        f"| **{total_bad}/{total_n}** "
        f"| **{overall_ratio:.2f}** "
        f"| **{sum_tc / max(total_n, 1):.2f}s** "
        f"| **{sum_tnc / max(total_n, 1):.2f}s** "
        f"| **{overall_speedup:.2f}x** |"
    )

    # Cache-regression details
    bad_lines = []
    for r in results:
        for row in r["rows"]:
            if row["bad"]:
                bad_lines.append(
                    f"- **{r['spec'].name}** [{row['i']}] {row['why']}\n"
                    f"  - prompt: `{row['prompt'][:150].replace(chr(10), ' ')}...`\n"
                    f"  - no-cache: `{row['no_cache'][:200]!r}`\n"
                    f"  - cache:    `{row['cache'][:200]!r}`"
                )
    if bad_lines:
        report_lines.append("\n### Cache regressions detected\n")
        report_lines.extend(bad_lines)
    else:
        report_lines.append("\n_No cache-only regressions detected._\n")

    report = "\n".join(report_lines) + "\n"
    print("\n" + "=" * 70)
    print(report)

    out_path = args.out or os.path.join(_REPO_ROOT, "scripts", "dream_cache_validation_report.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nReport written to: {out_path}")

    return 0 if total_bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
