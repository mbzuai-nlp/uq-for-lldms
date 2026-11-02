#!/usr/bin/env python
"""Bootstrap CIs for DiffusionGemma PRR: per-dataset + cross-dataset AGGREGATE.

Per-dataset: resample examples, normalized-PRR CI, paired Δ-vs-MSP.
Aggregate: each bootstrap iter resamples every dataset independently, averages each
method's PRR across datasets, and compares method-vs-MSP on that average. This is the
statistic behind the rebuttal claim "diffusion-native UQ ranks above MSP across task
types" — it pools the consistent-direction signal the single-dataset CIs can't.

Efficiency: oracle + random baseline depend only on the resampled quality, so compute
them ONCE per dataset-resample and reuse across methods.
"""
import sys, glob
import numpy as np
import torch
from lm_polygraph.ue_metrics.ue_metric import get_random_scores
from lm_polygraph.ue_metrics.pred_rej_area import PredictionRejectionArea

PRR = PredictionRejectionArea(0.5)
B = 2000
RND_ITERS = 200
MSP = "MaximumSequenceProbability"
TARGET_PREF = ["Comet", "AlignScore", "RougeL", "Accuracy"]   # Comet only exists for WMT19


def load(path):
    man = torch.load(path, weights_only=False, map_location="cpu")
    est, gm = man["estimations"], man["gen_metrics"]
    lvl = "sequence"
    cand = [g for (l, g) in gm.keys() if l == lvl]
    target = next((g for g in TARGET_PREF if g in cand and
                   np.nanstd(np.asarray(gm[(lvl, g)], float)) > 1e-9), None)
    q = np.asarray(gm[(lvl, target)], float)
    methods = sorted({m for (l, m) in est.keys() if l == lvl})
    U = {m: np.array([x.real if isinstance(x, complex) else x
                      for x in est[(lvl, m)]], float) for m in methods}
    name = path.split("/")[-4].split("',")[0].split("/")[-1]
    return dict(name=name, target=target, q=q, U=U, methods=methods, n=len(q))


def nprr_pt(ue, qq):
    val = PRR(ue, qq); orc = PRR(-qq, qq); rnd = get_random_scores(PRR, qq)
    d = orc - rnd
    return np.nan if abs(d) < 1e-12 else (val - rnd) / d


def main(paths):
    DS = [load(p) for p in paths]
    methods = DS[0]["methods"]
    RNG = np.random.default_rng(0)

    # point estimates
    point = {d["name"]: {m: nprr_pt(d["U"][m], d["q"]) for m in methods} for d in DS}

    # bootstrap: shared iter index, independent resample per dataset
    boot = {d["name"]: {m: np.empty(B) for m in methods} for d in DS}
    agg = {m: np.zeros(B) for m in methods}          # mean across datasets
    for d in DS:
        q = d["q"]; U = d["U"]; n = d["n"]; nm = d["name"]
        for b in range(B):
            ix = RNG.integers(0, n, n)
            qb = q[ix]; orc = PRR(-qb, qb); rnd = get_random_scores(PRR, qb, num_iter=RND_ITERS)
            den = orc - rnd
            for m in methods:
                v = np.nan if abs(den) < 1e-12 else (PRR(U[m][ix], qb) - rnd) / den
                boot[nm][m][b] = v
    for m in methods:
        agg[m] = np.nanmean([boot[d["name"]][m] for d in DS], axis=0)

    # ---- per-dataset ----
    for d in DS:
        nm = d["name"]
        order = sorted(methods, key=lambda m: -(point[nm][m] if np.isfinite(point[nm][m]) else -9))
        print(f"\n=== {nm}  (target={d['target']}, n={d['n']}, mean={d['q'].mean():.4f}, B={B}) ===")
        for m in order:
            lo, hi = np.nanpercentile(boot[nm][m], [2.5, 97.5])
            tag = "  <- MSP" if m == MSP else ""
            print(f"  {m:50s} {point[nm][m]:7.3f}  [{lo:6.3f},{hi:6.3f}]{tag}")

    # ---- aggregate across datasets ----
    pt_agg = {m: np.nanmean([point[d["name"]][m] for d in DS]) for m in methods}
    order = sorted(methods, key=lambda m: -pt_agg[m])
    print(f"\n=== AGGREGATE (mean normalized-PRR across {len(DS)} datasets) ===")
    print(f"  {'method':50s} {'meanPRR':>8s}  {'95% CI':>18s}")
    for m in order:
        lo, hi = np.nanpercentile(agg[m], [2.5, 97.5])
        tag = "  <- MSP" if m == MSP else ""
        print(f"  {m:50s} {pt_agg[m]:8.3f}  [{lo:6.3f},{hi:6.3f}]{tag}")
    print(f"\n  paired Δ(method - MSP) on the cross-dataset mean  (* = 95% CI excludes 0):")
    for m in order:
        if m == MSP:
            continue
        dd = agg[m] - agg[MSP]
        lo, hi = np.nanpercentile(dd, [2.5, 97.5])
        frac = np.nanmean(dd > 0)
        sig = "*" if (lo > 0 or hi < 0) else " "
        print(f"    {m:48s} Δ={pt_agg[m]-pt_agg[MSP]:+.3f} [{lo:+.3f},{hi:+.3f}] P(Δ>0)={frac:.2f} {sig}")


if __name__ == "__main__":
    paths = []
    for pat in sys.argv[1:]:
        paths += (sorted(glob.glob(pat)) if any(c in pat for c in "*?") else [pat])
    main(paths)
