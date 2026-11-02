#!/usr/bin/env python
"""Compare sampling-based UQ against trajectory-based UQ on DiffusionGemma.

The sampling estimators (SemanticEntropy, KLE, CoCoA-MSP) need K independent decodes per
input, so they cost about K times a single generation. The trajectory estimators read the
denoising trace of one generation and cost roughly +26% (see bench_dg_overhead.py). This
script puts both on one table so the price is visible next to the benefit.

Both sets are run over the same 100 examples per dataset, verified by comparing input_texts,
and each dataset is scored with its task-standard metric (AlignScore for SamSum and TriviaQA,
COMET for WMT19). Values are the harness-computed prr_0.5_normalized.

Usage:
    python compare_sampling_vs_trace.py <out_root> [csv_out]
"""
import os
import sys

import numpy as np
import torch
from scipy.stats import rankdata

UE = "prr_0.5_normalized"
LVL = "sequence"

# (label, trace-run dir, trace-run timestamp, sampling-run dir, quality metric)
DATASETS = [
    ("SamSum", "out/dg_samsum", "14-25-32", "out/dg_samsum_smp", "AlignScore"),
    ("TriviaQA", "out/dg_triviaqa", "15-05-41", "out/dg_triviaqa_smp", "AlignScore"),
    ("WMT19", "out/dg_wmt19", "15-21-59", "out/dg_wmt19_smp", "Comet"),
]

SAMPLING = {"SemanticEntropy", "KernelLanguageEntropy", "CocoaMSP"}

PRETTY = {
    "CocoaConfidenceAtCommitCrossEncoderBlockAvgTrace": "D-CoCoA-L",
    "KernelLanguageEntropyTrace": "KLE-trace",
    "LUQTrace": "LUQ-trace",
    "MaximumSequenceProbability": "MSP (baseline)",
    "KernelLanguageEntropy": "KLE (sampling)",
    "SemanticEntropy": "SemanticEntropy",
    "CocoaMSP": "CoCoA-MSP",
}


def find(root, ts=None):
    hits = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f == "ue_manager_seed1":
                p = os.path.join(dirpath, f)
                if ts is None or ts in p:
                    hits.append(p)
    return sorted(hits)


def main(root, csv_out=None):
    os.chdir(root)
    rows = {}
    for name, trace_dir, ts, samp_dir, metric in DATASETS:
        paths = [find(trace_dir, ts), find(samp_dir)]
        assert all(paths), f"missing pickles for {name}"
        # same-examples check: the comparison is only meaningful on identical inputs
        mans = [torch.load(p[0], weights_only=False, map_location="cpu") for p in paths]
        a, b = (list(m["stats"]["input_texts"]) for m in mans)
        if a != b:
            print(f"WARNING: {name} trace and sampling runs used different examples")
        for man in mans:
            for key, val in man["metrics"].items():
                if len(key) == 4 and key[0] == LVL and key[2] == metric and key[3] == UE:
                    rows.setdefault(key[1], {})[name] = val

    cols = [d[0] for d in DATASETS]
    table = [(m, [v[c] for c in cols], float(np.mean([v[c] for c in cols])))
             for m, v in rows.items() if len(v) == len(cols)]
    mat = np.array([r[1] for r in table])
    ranks = np.vstack([rankdata(-mat[:, j]) for j in range(mat.shape[1])]).T.mean(axis=1)
    order = np.argsort([-r[2] for r in table])

    header = f"{'Method':24s} {'cost':9s} " + " ".join(f"{c:>9s}" for c in cols) + f" {'Mean':>7s} {'Rank':>6s}"
    print(header)
    lines = ["Method,Cost," + ",".join(cols) + ",Mean,MeanRank"]
    for i in order:
        method, vals, mean = table[i]
        label = PRETTY.get(method, method)
        cost = f"{'~10x gen' if method in SAMPLING else '1x gen'}"
        print(f"{label:24s} {cost:9s} " + " ".join(f"{v:9.3f}" for v in vals)
              + f" {mean:7.3f} {ranks[i]:6.2f}")
        lines.append(f"{label},{cost}," + ",".join(f"{v:.3f}" for v in vals)
                     + f",{mean:.3f},{ranks[i]:.2f}")

    if csv_out:
        open(csv_out, "w").write("\n".join(lines) + "\n")
        print(f"\nCSV -> {csv_out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
