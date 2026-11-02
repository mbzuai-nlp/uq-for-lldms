import json, os, math, numpy as np
H = os.path.expanduser("~/dllm_demo/")
from sklearn.metrics import roc_auc_score
demo = {"gsm8k": [182, 599, 279], "triviaqa": [92, 235, 99], "coqa": [149, 146, 12]}
for ds in ["gsm8k", "triviaqa", "coqa"]:
    p = H + f"cocoa_{ds}.jsonl"
    if not os.path.exists(p): continue
    R = [json.loads(l) for l in open(p)]
    if ds == "gsm8k": S = {r["i"]: r for r in json.load(open(H + "scores.json"))}
    else: S = {r["i"]: r for r in json.load(open(H + f"scores_{ds}.json"))["res"]}
    v = np.array([r["cocoa"] for r in R]); lv = np.log(np.maximum(v, 1e-4))
    lo, hi = float(np.exp(np.percentile(lv, 1))), float(np.exp(np.percentile(lv, 99))); thr = float(np.percentile(v, 80))
    nz = lambda u: min(1, max(0, (math.log(max(u, 1e-4)) - math.log(lo)) / (math.log(hi) - math.log(lo))))
    y = [0 if S[r["i"]]["correct"] else 1 for r in R]
    dco = [S[r["i"]]["d_cocoa_l"] for r in R]
    auc = lambda s: round(roc_auc_score(y, s), 3) if 0 < sum(y) < len(y) else None
    print(f"{ds}: n={len(R)} lo={lo:.10g} hi={hi:.10g} thr={thr:.10g} thr_n={nz(thr):.2f}  AUC cocoa={auc(v)} d-cocoa-L={auc(dco)}  mean secs={np.mean([sum(r['secs'].values()) for r in R]):.1f}")
    for i in demo[ds]:
        r = next((r for r in R if r["i"] == i), None)
        if r: print(f"   id {i}: cocoa={r['cocoa']:.10g} norm={nz(r['cocoa']):.2f} correct={S[i]['correct']}")
