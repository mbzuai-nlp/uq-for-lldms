"""Record full denoising traces of LLaDA-1.5 on GSM8K for the landing-page GIF.

Decoding mirrors the paper setup (llada/generate.py::generate + get_transfer_index_dynamic,
factor=1, block_length=32, gen_length=256, greedy).
"""
import json, re, sys, time, os
import numpy as np, torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from llada.modeling_llada import LLaDAModelLM as AutoModel

MASK = 126336
DS = sys.argv[1]                      # triviaqa | coqa
N = int(sys.argv[2]); START = int(sys.argv[3]) if len(sys.argv) > 3 else 0
GEN, BLK = int(os.environ.get("GEN",20)), int(os.environ.get("GEN",20))                     # paper setup for QA: max_new_tokens=20 -> a single block
OUT = os.path.expanduser(os.environ.get("OUT", f"~/dllm_demo/traces_{DS}.jsonl"))

tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-1.5", trust_remote_code=True)
model = AutoModel.from_pretrained("GSAI-ML/LLaDA-1.5", trust_remote_code=True, torch_dtype=torch.bfloat16,
                                  device_map="auto", max_memory={0: "7GiB", 1: "7GiB", "cpu": "40GiB"}).eval()
dev = next(model.parameters()).device


def transfer_dynamic(conf_masked, factor=1.0):
    """number of tokens to commit (copy of get_transfer_index_dynamic)"""
    s = torch.sort(conf_masked, descending=True)[0]
    n = len(s)
    th = [1 - factor / (k + 2) for k in range(n)]
    th[0] = -1
    top_i = 0
    for top_i in range(n):
        if s[top_i] < th[top_i]:
            break
    if top_i == 0 or top_i == n - 1:
        top_i += 1
    return top_i


@torch.no_grad()
def run(prompt_ids):
    Lp = prompt_ids.shape[1]
    x = torch.full((1, Lp + GEN), MASK, dtype=torch.long, device=dev)
    x[:, :Lp] = prompt_ids
    steps = []
    commit_lp = [None] * GEN
    commit_step = [-1] * GEN
    g = 0
    for b in range(GEN // BLK):
        s, e = Lp + b * BLK, Lp + (b + 1) * BLK
        while (x[0, s:e] == MASK).any():
            logits = model(x).logits[:, Lp:]
            Lp_ = Lp; logits = logits.float()
            lpall = logits[0].log_softmax(-1)
            confall, amall = lpall.exp().max(-1)
            lp = lpall[b * BLK:(b + 1) * BLK]
            conf, am = confall[b * BLK:(b + 1) * BLK], amall[b * BLK:(b + 1) * BLK]
            maskb = x[0, s:e] == MASK
            k = transfer_dynamic(conf[maskb])
            c = torch.where(maskb, conf, torch.full_like(conf, -1.0))
            sel = torch.topk(c, k).indices
            before = x[0, Lp:].tolist()
            for j in sel.tolist():
                x[0, s + j] = am[j]
                commit_lp[b * BLK + j] = float(lp[j, am[j]])
                commit_step[b * BLK + j] = g
            steps.append(dict(block=b, state=before, argmax=amall.tolist(), conf=[round(v, 4) for v in confall.tolist()],
                              committed=[b * BLK + j for j in sel.tolist()]))
            g += 1
    return x[0, Lp:].tolist(), steps, commit_lp, commit_step


import string
def norm(t):
    t = t.lower().strip()
    t = "".join(c for c in t if c not in set(string.punctuation))
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    return " ".join(t.split())

def is_correct(pred, golds):
    p = norm(pred.split("\n")[0])
    if not p: return False
    for g in golds:
        g = norm(g)
        if g and (g == p or re.search(r"\b" + re.escape(g) + r"\b", p) or (len(p) > 2 and re.search(r"\b" + re.escape(p) + r"\b", g))):
            return True
    return False

name = {"triviaqa": "LM-Polygraph/triviaqa", "coqa": "LM-polygraph/coqa"}[DS]
ds = load_dataset(name, "simple_instruct", split="test")
with open(OUT, "a") as f:
    for i in range(START, START + N):
        q = ds[i]["input"]
        golds = ds[i]["output"] if isinstance(ds[i]["output"], list) else [ds[i]["output"]]
        ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                      tokenize=True, return_tensors="pt", return_dict=True)["input_ids"].to(dev)
        t0 = time.time()
        y, steps, clp, cst = run(ids)
        text = tok.decode(y, skip_special_tokens=True).strip()
        ok = is_correct(text, golds)
        rec = dict(i=i, ds=DS, question=q, gold=golds, text=text, pred=text.split("\n")[0], correct=ok, y=y,
                   steps=steps, commit_lp=clp, commit_step=cst, nfe=len(steps), secs=time.time() - t0)
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(i, len(steps), repr(text[:40]), golds[:2], ok, f"{time.time()-t0:.1f}s", flush=True)
