import json, re, os, sys, numpy as np, torch
from transformers import AutoTokenizer
from sentence_transformers import CrossEncoder

MASK, BLK = 126336, 32
tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-1.5", trust_remote_code=True)
ce = CrossEncoder("cross-encoder/stsb-roberta-large", device=sys.argv[1] if len(sys.argv) > 1 else "cpu")
special = set(tok.all_special_ids) | {MASK}

def dec(ids):
    return tok.decode([i for i in ids if i not in special], skip_special_tokens=True).strip()

def sims(pairs):
    out = [None] * len(pairs); todo = []
    for k, (a, b) in enumerate(pairs):
        if a == b: out[k] = 1.0
        elif not a or not b: out[k] = 0.0
        else: todo.append(k)
    if todo:
        s = ce.predict([pairs[k] for k in todo], batch_size=64, show_progress_bar=False)
        for k, v in zip(todo, s): out[k] = float(v)
    return out

def ans(t):
    m = re.search(r"answer is\s*\$?\s*(-?[\d,]*\.?\d+)", t)
    return m.group(1).replace(",", "") if m else None

def eq(a, b):
    try: return abs(float(a) - float(b)) < 1e-6
    except Exception: return a == b
res = []
for line in open(os.path.expanduser("~/dllm_demo/traces.jsonl")):
    r = json.loads(line)
    y = r["y"]; G = len(y); T = len(r["steps"])
    drafts = []
    for st in r["steps"]:
        d = [a if s == MASK else s for s, a in zip(st["state"], st["argmax"])]
        drafts.append(d)
    ytxt = dec(y)
    dtxt = [dec(d) for d in drafts]
    # full-trajectory dissimilarity to final + step-to-step
    s_full = sims([(t, ytxt) for t in dtxt])
    s_prev = sims([(dtxt[t - 1] if t else "", dtxt[t]) for t in range(T)])
    # AD-Block: per block trajectory vs final block text, only valid blocks
    blocks = sorted(set(st["block"] for st in r["steps"]))
    ad_blocks = {}
    for b in blocks:
        fin = dec(y[b * BLK:(b + 1) * BLK])
        if not fin: continue
        ts = [t for t, st in enumerate(r["steps"]) if st["block"] == b]
        ss = sims([(dec(drafts[t][b * BLK:(b + 1) * BLK]), fin) for t in ts])
        ad_blocks[b] = float(np.mean([1 - v for v in ss]))
    ad_block = float(np.mean(list(ad_blocks.values()))) if ad_blocks else 0.0
    valid = [k for k in range(G) if y[k] not in special and r["commit_lp"][k] is not None]
    commit_nll = float(np.mean([-r["commit_lp"][k] for k in valid])) if valid else 0.0
    ad_full = float(np.mean([1 - v for v in s_full]))
    w = [(T - t) / T for t in range(T)]  # earlier steps weighted more (progress weight t/T with t=T..1)
    wad_full = float(np.mean([wi * (1 - v) for wi, v in zip(w, s_full)]))
    res.append(dict(i=r["i"], correct=eq(r["pred"], r["gold"]), pred=r["pred"], gold=r["gold"], nfe=T,
                    commit_nll=commit_nll, ad_block=ad_block, ad_blocks=ad_blocks, ad_full=ad_full, wad_full=wad_full,
                    d_cocoa_l=commit_nll * ad_block, draft_ans=[ans(t) for t in dtxt],
                    d_full=[1 - v for v in s_full], d_prev=[1 - v for v in s_prev]))
    print(r["i"], T, r["correct"], round(commit_nll * ad_block, 4), [ans(t) for t in dtxt][:12], flush=True)
json.dump(res, open(os.path.expanduser("~/dllm_demo/scores.json"), "w"))
