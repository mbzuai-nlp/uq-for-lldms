"""QA version (TriviaQA / CoQA, single 20-token block): final scores for all traces + frame data for chosen ids.
usage: python viz_qa.py DS [id,id,...]"""
import json, re, os, sys, string, numpy as np
from transformers import AutoTokenizer
from sentence_transformers import CrossEncoder
MASK = 126336
DS = sys.argv[1]; want = set(map(int, sys.argv[2].split(","))) if len(sys.argv) > 2 else set()
tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-1.5", trust_remote_code=True)
ce = CrossEncoder("cross-encoder/stsb-roberta-large", device=os.environ.get("CE_DEV", "cuda:0"))
sp = set(tok.all_special_ids) | {MASK, 126348, 126081}
_c = {}
def dec(ids): return tok.decode([i for i in ids if i not in sp], skip_special_tokens=True).strip()
def sim(pairs):
    out, todo = [None]*len(pairs), []
    for k,(a,b) in enumerate(pairs):
        if (a,b) in _c: out[k]=_c[(a,b)]
        elif a==b: out[k]=1.0
        elif not a or not b: out[k]=0.0
        else: todo.append(k)
    if todo:
        s = ce.predict([pairs[k] for k in todo], batch_size=64, show_progress_bar=False)
        for k,v in zip(todo,s): out[k]=float(v); _c[pairs[k]]=float(v)
    return out
def norm(t):
    t = t.lower().strip(); t = "".join(c for c in t if c not in set(string.punctuation))
    t = re.sub(r"\b(a|an|the)\b", " ", t); return " ".join(t.split())
def ok(pred, golds):
    p = norm(pred.split("\n")[0])
    if not p: return False
    for g in golds:
        g = norm(g)
        if g and (g == p or re.search(r"\b"+re.escape(g)+r"\b", p) or (len(p) > 2 and re.search(r"\b"+re.escape(p)+r"\b", g))): return True
    return False
def split_prompt(q):
    tail = q[q.rfind("Question:"):]
    question = tail.removeprefix("Question:").split("\nAnswer:")[0].strip()
    ctx = q.split("\n\nAnswer the following question")[0].strip() if DS == "coqa" else None
    return question, ctx
res, viz = [], []
for line in open(os.path.expanduser(f"~/dllm_demo/traces_{DS}.jsonl")):
    r = json.loads(line); y, steps = r["y"], r["steps"]; T = len(steps)
    drafts = [[a if s==MASK else s for s,a in zip(st["state"], st["argmax"])] for st in steps]
    dt = [dec(d) for d in drafts]; yt = dec(y)
    L = next((k for k,v in enumerate(y) if v in sp), len(y))
    valid = [k for k in range(L) if r["commit_lp"][k] is not None]
    commit_nll = float(np.mean([-r["commit_lp"][k] for k in valid])) if valid else 0.0
    ad = float(np.mean([1-v for v in sim([(x, yt) for x in dt])])) if yt else 0.0   # single block: AD-Block == AD over the block
    res.append(dict(i=r["i"], correct=r["correct"], nfe=T, commit_nll=commit_nll, ad_block=ad, d_cocoa_l=commit_nll*ad,
                    draft=[x.split("\n")[0][:40] for x in dt], draft_ok=[ok(x, r["gold"]) for x in dt], pred=r["pred"], gold=r["gold"][:3]))
    if r["i"] in want:
        # running score on the prefix: provisional CommitNLL x AD vs current draft (exact at the last step)
        draft_nll, adr = [], []
        for t, st in enumerate(steps):
            vals = [-r["commit_lp"][k] if st["state"][k]!=MASK else -float(np.log(max(st["conf"][k],1e-6))) for k in range(L)]
            draft_nll.append(float(np.mean(vals)) if vals else 0.0)
            adr.append(float(np.mean([1-v for v in sim([(dt[s], dt[t]) for s in range(t+1)])])) if dt[t] else 1.0)
        # the last frame shows the state *before* the final commit; make the last running value exact
        adr[-1] = ad; draft_nll[-1] = commit_nll
        frames = [dict(state=[None if s==MASK else s for s in st["state"][:L]], ghost=st["argmax"][:L], conf=st["conf"][:L],
                       commit=[k for k in st["committed"] if k<L], block=0, ans=dt[t].split("\n")[0][:40] or None, ok=ok(dt[t], r["gold"]))
                  for t, st in enumerate(steps)]
        vocab = sorted(set(i for f in frames for i in f["ghost"]+[x for x in f["state"] if x is not None]) | set(y[:L]))
        qn, ctx = split_prompt(r["question"])
        viz.append(dict(i=r["i"], ds=DS, question=qn, context=ctx, gold=r["gold"][0], golds=r["gold"], pred=r["pred"], pred_ok=r["correct"],
                        L=L, T=T, y=y[:L], commit_lp=[r["commit_lp"][k] for k in range(L)], vocab={i: tok.decode([i]) for i in vocab}, frames=frames,
                        sig=dict(score=[None]+[draft_nll[t]*adr[t] for t in range(1,T)], comp1=[None]+adr[1:], comp2=draft_nll)))
v = np.array([x["d_cocoa_l"] for x in res]); thr = float(np.percentile(v, 80))
for x in res: x["pct"] = int(round((v < x["d_cocoa_l"]).mean()*100))
json.dump(dict(thr=thr, res=res), open(os.path.expanduser(f"~/dllm_demo/scores_{DS}.json"), "w"))
if viz:
    for d in viz: d["pct_final"] = next(x["pct"] for x in res if x["i"]==d["i"])
    json.dump(dict(thr=thr, viz=viz), open(os.path.expanduser(f"~/dllm_demo/viz_{DS}.json"), "w"))
from sklearn.metrics import roc_auc_score
yb = [0 if x["correct"] else 1 for x in res]
print(DS, "n", len(res), "acc", np.mean([x["correct"] for x in res]).round(3), "thr", round(thr,4),
      "AUC d_cocoa_l", round(roc_auc_score(yb, v),3) if 0<sum(yb)<len(yb) else "-")
