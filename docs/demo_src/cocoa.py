"""Standard CoCoA-MCNLL for recorded traces (for normalisation + the demo's comparison card).
usage: python cocoa.py DS N_FIRST id,id,...   (DS: gsm8k | triviaqa | coqa)
CoCoA-MCNLL = MCNLL(y) * mean_i (1 - sim(y, y_i)), y = greedy answer, y_i = M temperature-1 samples."""
import json, os, sys, time, math, numpy as np, torch
from transformers import AutoTokenizer
from llada.modeling_llada import LLaDAModelLM
from sentence_transformers import CrossEncoder
MASK = 126336; M = 5; NMC = 16
DS, NFIRST = sys.argv[1], int(sys.argv[2]); EXTRA = set(map(int, sys.argv[3].split(","))) if len(sys.argv) > 3 else set()
GEN, BLK = (256, 32) if DS == "gsm8k" else (20, 20)
src = os.path.expanduser("~/dllm_demo/" + ("traces.jsonl" if DS == "gsm8k" else f"traces_{DS}.jsonl"))
out = os.path.expanduser(f"~/dllm_demo/cocoa_{DS}.jsonl")
done = set(json.loads(l)["i"] for l in open(out)) if os.path.exists(out) else set()
tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-1.5", trust_remote_code=True)
model = LLaDAModelLM.from_pretrained("GSAI-ML/LLaDA-1.5", trust_remote_code=True, torch_dtype=torch.bfloat16,
                                     device_map="auto", max_memory={0: "7GiB", 1: "7GiB", "cpu": "40GiB"}).eval()
dev = next(model.parameters()).device
ce = CrossEncoder("cross-encoder/stsb-roberta-large", device="cpu")
sp = set(tok.all_special_ids) | {MASK, 126348, 126081}
def dec(ids): return tok.decode([i for i in ids if i not in sp], skip_special_tokens=True).strip()

def n_commit(conf):
    s = torch.sort(conf, descending=True)[0]; n = len(s)
    th = [1 - 1.0 / (k + 2) for k in range(n)]; th[0] = -1; top = 0
    for top in range(n):
        if s[top] < th[top]: break
    if top == 0 or top == n - 1: top += 1
    return top

@torch.no_grad()
def sample(ids, g):
    Lp = ids.shape[1]; x = torch.full((1, Lp + GEN), MASK, dtype=torch.long, device=dev); x[:, :Lp] = ids
    for b in range(GEN // BLK):
        s, e = Lp + b * BLK, Lp + (b + 1) * BLK
        while (x[0, s:e] == MASK).any():
            lg = model(x).logits[0, s:e].float()
            p = lg.softmax(-1)
            gum = -torch.log(-torch.log(torch.rand(lg.shape, generator=g, device=lg.device).clamp_min(1e-20)))
            x0 = (lg + gum).argmax(-1)                      # temperature-1 sample (Gumbel-max)
            conf = p.gather(-1, x0[:, None])[:, 0]
            m = x[0, s:e] == MASK
            c = torch.where(m, conf, torch.full_like(conf, -1.0))
            sel = torch.topk(c, n_commit(conf[m])).indices
            x[0, s + sel] = x0[sel]
    return x[0, Lp:].tolist()

@torch.no_grad()
def mcnll(ids, y, g):
    L = len(y)
    if L == 0: return 0.0
    seq = torch.cat([ids[0], torch.tensor(y, device=dev)])[None]; Lp = ids.shape[1]; tot = []
    bs = 2 if DS == "gsm8k" else 4
    for r in range(NMC // bs):
        batch = seq.repeat(bs, 1); ls = []
        for k in range(bs):
            l = int(torch.randint(1, L + 1, (1,), generator=g, device=dev)); ls.append(l)
            pos = torch.randperm(L, generator=g, device=dev)[:l] + Lp
            batch[k, pos] = MASK
        lp = model(batch).logits[:, Lp:].float().log_softmax(-1)
        for k in range(bs):
            msk = batch[k, Lp:] == MASK
            nll = -lp[k][msk].gather(-1, torch.tensor(y, device=dev)[msk][:, None]).sum()
            tot.append(float(nll) * L / ls[k])
    return float(np.mean(tot))

f = open(out, "a")
for line in open(src):
    r = json.loads(line)
    if r["i"] in done: continue
    if not (r["i"] in EXTRA or len(done) < NFIRST): continue
    ids = tok.apply_chat_template([{"role": "user", "content": r["question"]}], add_generation_prompt=True,
                                  tokenize=True, return_tensors="pt", return_dict=True)["input_ids"].to(dev)
    y = r["y"]; L = next((k for k, v in enumerate(y) if v in sp), len(y)); yc = y[:L]
    g = torch.Generator(device=dev); g.manual_seed(1000 + r["i"])
    t0 = time.time(); u_inf = mcnll(ids, yc, g); t1 = time.time()
    samples = [dec(sample(ids, g)) for _ in range(M)]; t2 = time.time()
    yt = dec(y)
    sims = ce.predict([(yt, s) for s in samples], show_progress_bar=False) if yt else np.zeros(M)
    cons = float(np.mean([1 - float(v) for v in sims])); t3 = time.time()
    rec = dict(i=r["i"], mcnll=u_inf, cons=cons, cocoa=u_inf * cons, samples=samples, secs=dict(mcnll=t1 - t0, sampling=t2 - t1, ce=t3 - t2))
    f.write(json.dumps(rec) + "\n"); f.flush(); done.add(r["i"])
    print(r["i"], round(u_inf, 3), round(cons, 3), round(u_inf * cons, 4), [s[:25] for s in samples], {k: round(v, 1) for k, v in rec["secs"].items()}, flush=True)
