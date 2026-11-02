"""Build compact per-frame data for the GIF from recorded traces (ids given on argv)."""
import json, re, os, sys, numpy as np
from transformers import AutoTokenizer
from sentence_transformers import CrossEncoder
MASK, BLK = 126336, 32
tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-1.5", trust_remote_code=True)
ce = CrossEncoder("cross-encoder/stsb-roberta-large", device=os.environ.get("CE_DEV", "cpu"))
sp = set(tok.all_special_ids) | {MASK}
_cache = {}
def dec(ids): return tok.decode([i for i in ids if i not in sp], skip_special_tokens=True).strip()
def sim(pairs):
    out, todo = [None]*len(pairs), []
    for k,(a,b) in enumerate(pairs):
        if (a,b) in _cache: out[k]=_cache[(a,b)]
        elif a==b: out[k]=1.0
        elif not a or not b: out[k]=0.0
        else: todo.append(k)
    if todo:
        s = ce.predict([pairs[k] for k in todo], batch_size=32, show_progress_bar=False)
        for k,v in zip(todo,s): out[k]=float(v); _cache[pairs[k]]=float(v)
    return out
def ans(t):
    m = re.search(r"answer is\s*\$?\s*(-?[\d,]*\.?\d+)", t)
    return m.group(1).replace(",", "") if m else None
def eq(a, b):
    try: return abs(float(a) - float(b)) < 1e-6
    except Exception: return a == b
want = set(map(int, sys.argv[1].split(",")))
out = []
for line in open(os.path.expanduser("~/dllm_demo/traces.jsonl")):
    r = json.loads(line)
    if r["i"] not in want: continue
    y, steps = r["y"], r["steps"]; T = len(steps)
    L = next((k for k,v in enumerate(y) if v in sp), len(y))  # visible length = up to first special
    drafts = [[a if s==MASK else s for s,a in zip(st["state"], st["argmax"])] for st in steps]
    dt = [dec(d) for d in drafts]; yt = dec(y)
    d_final = [1-v for v in sim([(x, yt) for x in dt])]
    d_prev = [0.0] + [1-v for v in sim([(dt[t-1], dt[t]) for t in range(1,T)])]
    ad_run = [float(np.mean([1-v for v in sim([(dt[s], dt[t]) for s in range(t+1)])])) for t in range(T)]
    # per-step provisional NLL of the draft over visible positions
    draft_nll, commit_run = [], []
    lp_commit = r["commit_lp"]
    for t, st in enumerate(steps):
        vals=[]
        for k in range(L):
            if st["state"][k]!=MASK: vals.append(-lp_commit[k])
            else: vals.append(-float(np.log(max(st["conf"][k],1e-6))))
        draft_nll.append(float(np.mean(vals)))
        com=[-lp_commit[k] for k in range(L) if steps[t]["state"][k]!=MASK or k in st["committed"]]
        commit_run.append(float(np.mean(com)) if com else None)
    # paper-style AD-Block computed on the prefix (current block's provisional final = its current draft)
    adb_run=[]
    for t in range(T):
        per=[]
        for b in sorted(set(s["block"] for s in steps[:t+1])):
            ts=[s for s in range(t+1) if steps[s]["block"]==b]
            fin = dec(y[b*BLK:(b+1)*BLK]) if steps[t]["block"]>b else dec(drafts[t][b*BLK:(b+1)*BLK])
            if not fin: continue
            per.append(np.mean([1-v for v in sim([(dec(drafts[s][b*BLK:(b+1)*BLK]), fin) for s in ts])]))
        adb_run.append(float(np.mean(per)) if per else 0.0)
    frames=[]
    for t, st in enumerate(steps):
        frames.append(dict(state=[None if s==MASK else s for s in st["state"][:L]], ghost=st["argmax"][:L],
                           conf=st["conf"][:L], commit=[k for k in st["committed"] if k<L], block=st["block"],
                           ans=ans(dt[t])))
    vocab = sorted(set(i for f in frames for i in (f["ghost"]+[x for x in f["state"] if x is not None])) | set(y[:L]))
    out.append(dict(i=r["i"], question=r["question"].split("Now answer the following question.")[-1].strip().removeprefix("Question: ").removesuffix("Answer:").strip(),
                    gold=r["gold"], pred=r["pred"], correct=eq(r["pred"], r["gold"]), L=L, T=T, y=y[:L],
                    commit_lp=[lp_commit[k] for k in range(L)], commit_step=r["commit_step"][:L],
                    vocab={i: tok.decode([i]) for i in vocab}, frames=frames,
                    sig=dict(score=[None]+[draft_nll[t]*adb_run[t] for t in range(1,T)], comp1=[None]+adb_run[1:], comp2=draft_nll, d_final=d_final, d_prev=d_prev, ad_run=ad_run, draft_nll=draft_nll, commit_run=commit_run, adb_run=adb_run)))
    print(r["i"], "T",T,"L",L, flush=True)
    for k in ["score","d_final","d_prev","ad_run","draft_nll","commit_run","adb_run"]:
        print(k, " ".join(f"{v:.2f}" if v is not None else "-" for v in out[-1]["sig"][k]))
json.dump(out, open(os.path.expanduser(f"~/dllm_demo/viz_{sys.argv[1].replace(',','_')}.json"), "w"))
