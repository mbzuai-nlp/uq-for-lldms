#!/usr/bin/env python
"""Report how often DiffusionGemma's K decodes collapse to one string, from a sweep log.

Sampling-based UQ (SemanticEntropy, KLE, CoCoA-MSP) only has signal where repeated decodes
of a prompt differ. DiffusionGemma has no multinomial sampler; variation comes from the
random canvas initialisation and renoising, so how much diversity there is depends on the
model and the dataset rather than on a setting.

A prompt whose K decodes agree is not a failure: the model is confident there and that is
the calibrated answer. What matters is the corpus-level rate, because a corpus that
collapses everywhere leaves the semantic estimators nothing to measure.

This reads the sweep log rather than the pickle because the harness persists only
greedy_texts / greedy_tokens / input_texts / target_texts, dropping sample_texts. Sweeps run
at batch_size=1, so each collapse warning corresponds to exactly one input.

Usage:
    python sample_diversity.py <sweep_log.out> [...]
"""
import re
import sys
import glob

COLLAPSE_OLD = "returned identical decodes for every input"
COLLAPSE_NEW = re.compile(r"(\d+)/(\d+) inputs returned K identical decodes")
N_INPUTS = re.compile(r"subsample_eval_dataset=(\d+)")


def main(path):
    text = open(path, errors="replace").read()

    n_total = None
    m = N_INPUTS.search(text)
    if m:
        n_total = int(m.group(1))

    # Newer calculator reports the rate directly.
    m = COLLAPSE_NEW.search(text)
    if m:
        collapsed, seen = int(m.group(1)), int(m.group(2))
        print(f"{path}\n  reported: {collapsed}/{seen} inputs collapsed "
              f"({100.0 * collapsed / seen:.1f}%)")
        return

    # Older calculator warned once per collapsed input; batch_size=1 makes that a count.
    collapsed = text.count(COLLAPSE_OLD)
    if collapsed == 0:
        print(f"{path}\n  no collapse warnings found "
              "(either every input produced varied decodes, or this is not a sampling run)")
        return
    if n_total:
        print(f"{path}\n  {collapsed}/{n_total} inputs collapsed to a single decode "
              f"({100.0 * collapsed / n_total:.1f}%)")
        print(f"  {n_total - collapsed} inputs produced varied decodes, which is where the "
              "semantic estimators get their signal")
    else:
        print(f"{path}\n  {collapsed} inputs collapsed (total unknown)")


if __name__ == "__main__":
    for pat in sys.argv[1:]:
        for p in (sorted(glob.glob(pat)) if any(c in pat for c in "*?") else [pat]):
            try:
                main(p)
            except Exception:
                import traceback
                traceback.print_exc()
