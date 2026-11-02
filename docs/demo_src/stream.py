"""Time EXACTLY the live D-CoCoA-L update the demo shows after every denoising step (same maths as viz.py /
viz_qa.py, computed incrementally the way an online implementation would: cross-encoder pairs whose result
cannot change are cached, the rest are re-scored in one batch per step).
  GSM8K (blocks of 32): AD-Block over the blocks so far; the current block is compared with its current draft,
                        finished blocks with their final text (scored once, when the block finishes).
  QA (one block):       AD of all drafts so far vs the current draft.
  plus the provisional CommitNLL of the draft.
usage: python stream.py [repeats]      -> $BENCH_DIR/stream.json (per-step ms, median over repeats)"""
import json, os, sys, time, numpy as np, torch
from transformers import AutoTokenizer
from sentence_transformers import CrossEncoder
MASK = 126336
W = os.environ["BENCH_DIR"] + "/"; REP = int(sys.argv[1]) if len(sys.argv) > 1 else 3
EX = [("gsm8k", 1046), ("gsm8k", 599), ("gsm8k", 279), ("triviaqa", 92), ("triviaqa", 20), ("triviaqa", 99),
      ("coqa", 50), ("coqa", 146), ("coqa", 12)]
tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-1.5", trust_remote_code=True)
ce = CrossEncoder("cross-encoder/stsb-roberta-large", device="cuda:0")
sp = set(tok.all_special_ids) | {MASK, 126348, 126081}
def dec(ids): return tok.decode([i for i in ids if i not in sp], skip_special_tokens=True).strip()
def sync(): torch.cuda.synchronize()

def load(ds, i):
    for l in open(W + ("traces.jsonl" if ds == "gsm8k" else f"traces_{ds}.jsonl")):
        r = json.loads(l)
        if r["i"] == i: return r

def run(r, BLK):
    """returns per-step wall-clock (ms) of the live update"""
    y, steps = r["y"], r["steps"]; L = next((k for k, v in enumerate(y) if v in sp), len(y)); clp = r["commit_lp"]
    cache, out = {}, []
    def sims(pairs):
        todo = [p for p in pairs if p not in cache and p[0] != p[1] and p[0] and p[1]]
        todo = list(dict.fromkeys(todo))
        if todo:
            for p, v in zip(todo, ce.predict(todo, batch_size=64, show_progress_bar=False)): cache[p] = float(v)
        return [1.0 if a == b else 0.0 if not a or not b else cache[(a, b)] for a, b in pairs]
    drafts = []
    for t, st in enumerate(steps):
        sync(); t0 = time.perf_counter()
        d = [a if s == MASK else s for s, a in zip(st["state"], st["argmax"])]; drafts.append(d)
        vals = [-clp[k] if st["state"][k] != MASK else -float(np.log(max(st["conf"][k], 1e-6))) for k in range(L)]
        nll = float(np.mean(vals)) if vals else 0.0
        if BLK is None:      # QA: one block, compare every draft so far with the current draft
            txt = [dec(x) for x in drafts[-1:]]
            if t == 0: run.txt = []
            run.txt.append(txt[0])
            ad = float(np.mean([1 - v for v in sims([(s, run.txt[-1]) for s in run.txt])]))
        else:                # GSM8K: AD-Block on the prefix
            per = []
            for b in sorted(set(s["block"] for s in steps[:t + 1])):
                ts = [s for s in range(t + 1) if steps[s]["block"] == b]
                fin = dec(y[b * BLK:(b + 1) * BLK]) if st["block"] > b else dec(d[b * BLK:(b + 1) * BLK])
                if not fin: continue
                per.append(np.mean([1 - v for v in sims([(dec(drafts[s][b * BLK:(b + 1) * BLK]), fin) for s in ts])]))
            ad = float(np.mean(per)) if per else 0.0
        score = nll * ad
        sync(); out.append(1000 * (time.perf_counter() - t0))
    return out

ce.predict([("warm", "up")] * 8, show_progress_bar=False)
res = {}
for ds, i in EX:
    r = load(ds, i); BLK = 32 if ds == "gsm8k" else None
    reps = [run(r, BLK) for _ in range(REP)]
    ms = [float(np.median([rp[t] for rp in reps])) for t in range(len(reps[0]))]
    res[f"{ds}_{i}"] = dict(nfe=len(ms), step_ms=[round(v, 2) for v in ms], mean_ms=float(np.mean(ms)), max_ms=float(np.max(ms)))
    print(ds, i, "nfe", len(ms), f"mean {np.mean(ms):.1f} ms  median {np.median(ms):.1f}  max {np.max(ms):.1f}", flush=True)
json.dump(dict(device=torch.cuda.get_device_name(0), repeats=REP, res=res), open(W + "stream.json", "w"), indent=1)
print("saved")
