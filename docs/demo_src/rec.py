"""Re-record every trace of the demo pools on the benchmark GPU (same decoding as trace.py / trace_qa.py).
usage: python rec.py DS          (reads the old traces only for the prompts / gold answers)
writes $W/traces*.jsonl in the original format, timed with cuda sync."""
import json, os, re, sys, time, string, torch
from transformers import AutoTokenizer
from llada.modeling_llada import LLaDAModelLM

MASK = 126336
W = os.environ["BENCH_DIR"]; DS = sys.argv[1]
GEN, BLK = (256, 32) if DS == "gsm8k" else (20, 20)
name = "traces.jsonl" if DS == "gsm8k" else f"traces_{DS}.jsonl"
src = [json.loads(l) for l in open(os.path.join(W, "old", name))]
tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-1.5", trust_remote_code=True)
model = LLaDAModelLM.from_pretrained("GSAI-ML/LLaDA-1.5", trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda:0").eval()
dev = "cuda:0"

def n_commit(conf):
    s = torch.sort(conf, descending=True)[0]; n = len(s)
    th = [1 - 1.0 / (k + 2) for k in range(n)]; th[0] = -1; top = 0
    for top in range(n):
        if s[top] < th[top]: break
    if top == 0 or top == n - 1: top += 1
    return top

@torch.no_grad()
def run(ids):
    Lp = ids.shape[1]; x = torch.full((1, Lp + GEN), MASK, dtype=torch.long, device=dev); x[:, :Lp] = ids
    steps, clp, cst, g = [], [None] * GEN, [-1] * GEN, 0
    for b in range(GEN // BLK):
        s, e = Lp + b * BLK, Lp + (b + 1) * BLK
        while (x[0, s:e] == MASK).any():
            lg = model(x).logits[0, Lp:]
            lp = lg[b * BLK:(b + 1) * BLK].float().log_softmax(-1); conf, am = lp.exp().max(-1)
            m = x[0, s:e] == MASK
            c = torch.where(m, conf, torch.full_like(conf, -1.0))
            sel = torch.topk(c, n_commit(conf[m])).indices
            steps.append(dict(block=b, state=x[0, Lp:].tolist(), argmax=lg.argmax(-1).tolist(),
                              conf=[round(v, 4) for v in lg.float().softmax(-1).max(-1)[0].tolist()],
                              committed=[b * BLK + j for j in sel.tolist()]))
            for j in sel.tolist(): clp[b * BLK + j] = float(lp[j, am[j]]); cst[b * BLK + j] = g
            x[0, s + sel] = am[sel]; g += 1
    return x[0, Lp:].tolist(), steps, clp, cst

def num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
    return m[-1].replace(",", "").rstrip(".") if m else None
def norm(t):
    t = t.lower().strip(); t = "".join(c for c in t if c not in set(string.punctuation))
    t = re.sub(r"\b(a|an|the)\b", " ", t); return " ".join(t.split())
def is_correct(pred, golds):
    p = norm(pred.split("\n")[0])
    if not p: return False
    for g in golds:
        g = norm(g)
        if g and (g == p or re.search(r"\b" + re.escape(g) + r"\b", p) or (len(p) > 2 and re.search(r"\b" + re.escape(p) + r"\b", g))):
            return True
    return False

with open(os.path.join(W, name), "w") as f:
    for r in src:
        ids = tok.apply_chat_template([{"role": "user", "content": r["question"]}], add_generation_prompt=True,
                                      tokenize=True, return_tensors="pt", return_dict=True)["input_ids"].to(dev)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        y, steps, clp, cst = run(ids)
        torch.cuda.synchronize(); secs = time.perf_counter() - t0
        text = tok.decode(y, skip_special_tokens=True)
        if DS == "gsm8k":
            pred = num(text.split("The answer is")[-1]) if "The answer is" in text else num(text)
            rec = dict(i=r["i"], question=r["question"], gold=r["gold"], text=text, pred=pred, correct=(pred == r["gold"]))
        else:
            text = text.strip()
            rec = dict(i=r["i"], ds=DS, question=r["question"], gold=r["gold"], text=text, pred=text.split("\n")[0],
                       correct=is_correct(text, r["gold"]))
        rec.update(y=y, steps=steps, commit_lp=clp, commit_step=cst, nfe=len(steps), secs=secs, same_as_old=(y == r["y"]))
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(DS, r["i"], len(steps), rec["correct"], rec["same_as_old"], f"{secs:.2f}s", flush=True)
