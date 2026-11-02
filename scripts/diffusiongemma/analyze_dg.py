#!/usr/bin/env python
"""Analyze a DiffusionGemma ue_manager dict (torch.save): finite counts + PRR.

Reads the harness-computed prr_0.5_normalized from man["metrics"] (authoritative),
and independently recomputes it with lm_polygraph's PredictionRejectionArea so we
can compute PRR against ANY quality metric present (e.g. a post-hoc BLEU for wmt19).
"""
import sys, glob
import numpy as np
import torch

from lm_polygraph.ue_metrics.ue_metric import get_random_scores
from lm_polygraph.ue_metrics.pred_rej_area import PredictionRejectionArea

PRR = PredictionRejectionArea(0.5)


def norm_prr(ue, quality):
    ue = np.array([x.real if isinstance(x, complex) else x for x in ue], float)
    quality = np.asarray(quality, float)
    m = np.isfinite(ue) & np.isfinite(quality)
    ue, quality = ue[m], quality[m]
    if len(ue) < 3:
        return float("nan")
    val = PRR(ue, quality)
    oracle = PRR(-quality, quality)
    rand = get_random_scores(PRR, quality)
    if abs(oracle - rand) < 1e-12:
        return float("nan")
    return (val - rand) / (oracle - rand)


def main(path):
    man = torch.load(path, weights_only=False, map_location="cpu")
    print("===", path)
    metrics = man["metrics"]
    est = man["estimations"]
    gm = man["gen_metrics"]
    level = "sequence"

    gm_names = sorted({k[1] for k in gm.keys() if k[0] == level})
    print("gen_metrics present:", gm_names)
    for g in gm_names:
        v = np.asarray(gm[(level, g)], float)
        print(f"  quality '{g}': n={len(v)} finite={np.isfinite(v).sum()} mean={np.nanmean(v):.4f}")

    methods = sorted({k[1] for k in est.keys() if k[0] == level})

    # finite counts per estimator
    print("\nfinite counts per estimator:")
    for mth in methods:
        v = np.array([x.real if isinstance(x, complex) else x for x in est[(level, mth)]], float)
        nfin = np.isfinite(v).sum()
        flag = "" if nfin == len(v) else "  <-- NON-FINITE"
        print(f"  {mth:52s} {nfin:>4d}/{len(v):<4d}{flag}")

    # PRR table for each quality metric
    for g in gm_names:
        q = np.asarray(gm[(level, g)], float)
        rows = []
        for mth in methods:
            stored = metrics.get((level, mth, g, "prr_0.5_normalized"), None)
            recomputed = norm_prr(est[(level, mth)], q)
            rows.append((mth, stored, recomputed))
        rows.sort(key=lambda r: -(r[1] if (r[1] is not None and np.isfinite(r[1])) else
                                  (r[2] if np.isfinite(r[2]) else -9)))
        print(f"\n--- PRR_0.5_normalized vs quality='{g}'  (stored | recomputed) ---")
        for mth, stored, recomp in rows:
            s = f"{stored:8.4f}" if (stored is not None and np.isfinite(stored)) else "   ---  "
            print(f"  {mth:52s} {s} | {recomp:8.4f}")
    print()


if __name__ == "__main__":
    for pat in sys.argv[1:]:
        matches = sorted(glob.glob(pat)) if any(c in pat for c in "*?") else [pat]
        for p in matches:
            try:
                main(p)
            except Exception:
                import traceback; traceback.print_exc()
