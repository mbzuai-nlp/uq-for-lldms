#!/usr/bin/env python
"""Consolidate the 3 DiffusionGemma pickles into one wide PRR table (methods x
datasets) with Mean and Mean-Rank, in the paper's build_*_table format.
Each dataset uses its appropriate target quality metric.
Emits CSV to stdout-adjacent file and prints a readable table.
"""
import numpy as np, torch
from scipy.stats import rankdata

# (label, pickle path, target gen-metric)
DS = [
    ("SamSum/AlignScore",  "SAMSUM",   "AlignScore"),
    ("TriviaQA/AlignScore","TRIVIAQA", "AlignScore"),
    ("WMT19/Comet",        "WMT19",    "Comet"),
]
PATHS = {
    "SAMSUM":   __import__("sys").argv[1],
    "TRIVIAQA": __import__("sys").argv[2],
    "WMT19":    __import__("sys").argv[3],
}
OUT = __import__("sys").argv[4]

FRIENDLY = {
    "MaximumSequenceProbability": "MSP (baseline)",
    "TraceEntropy_avg": "TraceEntropy_avg",
    "TraceEntropy_max": "TraceEntropy_max",
    "TraceProbs_avg": "TraceProbs_avg",
    "TraceProbs_max": "TraceProbs_max",
    "ConfidenceAtCommit_avg": "ConfidenceAtCommit_avg",
    "ConfidenceAtCommit_min": "ConfidenceAtCommit_min",
    "KernelLanguageEntropyTrace": "KLE-trace",
    "LUQTrace": "LUQ-trace",
    "CocoaConfidenceAtCommitCrossEncoderBlockAvgTrace": "D-CoCoA-L",
}
UE = "prr_0.5_normalized"
LVL = "sequence"

cols = {}
for label, key, tgt in DS:
    man = torch.load(PATHS[key], weights_only=False, map_location="cpu")
    m = man["metrics"]
    row = {}
    for (l, meth, gmet, ue), v in m.items():
        if l == LVL and gmet == tgt and ue == UE:
            row[meth] = v
    cols[label] = row

methods = list(FRIENDLY.keys())
labels = [d[0] for d in DS]

# build matrix
M = np.full((len(methods), len(labels)), np.nan)
for j, lab in enumerate(labels):
    for i, meth in enumerate(methods):
        M[i, j] = cols[lab].get(meth, np.nan)

mean = np.nanmean(M, axis=1)
# rank per column (1=best=highest PRR)
ranks = np.zeros_like(M)
for j in range(M.shape[1]):
    ranks[:, j] = rankdata(-M[:, j])
mean_rank = ranks.mean(axis=1)

# order by mean PRR desc
order = np.argsort(-mean)

# print + CSV
hdr = ["Method"] + labels + ["Mean", "MeanRank"]
lines = [",".join(hdr)]
print(f"{'Method':24s} " + " ".join(f"{l:20s}" for l in labels) + f" {'Mean':>7s} {'Rank':>6s}")
for i in order:
    name = FRIENDLY[methods[i]]
    vals = [f"{M[i,j]:.3f}" for j in range(len(labels))]
    lines.append(",".join([name] + vals + [f"{mean[i]:.3f}", f"{mean_rank[i]:.2f}"]))
    star = "  *baseline*" if methods[i] == "MaximumSequenceProbability" else ""
    print(f"{name:24s} " + " ".join(f"{M[i,j]:>20.3f}" for j in range(len(labels))) +
          f" {mean[i]:7.3f} {mean_rank[i]:6.2f}{star}")

open(OUT, "w").write("\n".join(lines) + "\n")
print(f"\nCSV -> {OUT}")
# also print target-metric means
for label, key, tgt in DS:
    man = torch.load(PATHS[key], weights_only=False, map_location="cpu")
    q = np.asarray(man["gen_metrics"][(LVL, tgt)], float)
    print(f"  {label}: mean {tgt}={np.nanmean(q):.4f} (n={len(q)})")
