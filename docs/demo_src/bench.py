"""Wall-clock benchmark of the 9 landing-page demo examples on ONE GPU (no CPU offload).

Measures, per example:
  gen      : greedy LLaDA-1.5 decoding (same schedule as trace.py / trace_qa.py), time per denoising step
  dcocoa   : D-CoCoA-L after generation = CommitNLL (free) + AD-Block (cross-encoder on per-block drafts)
  stream   : cost of updating the live score after one step (one cross-encoder pair)
  cocoa    : standard CoCoA-MCNLL = MCNLL (16 masked passes) + 5 temperature-1 samples + cross-encoder
usage: python bench.py [cuda:N] [repeats]
"""
import json, os, sys, time, numpy as np, torch
from transformers import AutoTokenizer
from llada.modeling_llada import LLaDAModelLM
from sentence_transformers import CrossEncoder

MASK, M, NMC = 126336, 5, 16
H = os.environ.get("BENCH_DIR", os.path.expanduser("~/dllm_demo")) + "/"
EX = [("gsm8k", 1046), ("gsm8k", 599), ("gsm8k", 279), ("triviaqa", 92), ("triviaqa", 20), ("triviaqa", 99),
      ("coqa", 50), ("coqa", 146), ("coqa", 12)]
REP = int(sys.argv[2]) if len(sys.argv) > 2 else 3

if len(sys.argv) > 1: dev = sys.argv[1]
else:
    free = [torch.cuda.mem_get_info(k)[0] for k in range(torch.cuda.device_count())]
    dev = f"cuda:{int(np.argmax(free))}"
free = torch.cuda.mem_get_info(torch.device(dev))[0] / 2**30
print("device", dev, torch.cuda.get_device_name(dev), f"free {free:.1f} GiB", flush=True)
if free < 20: sys.exit("need >= 20 GiB free on one GPU (model 16 GiB + cross-encoder + activations)")

tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-1.5", trust_remote_code=True)
model = LLaDAModelLM.from_pretrained("GSAI-ML/LLaDA-1.5", trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev).eval()
ce = CrossEncoder("cross-encoder/stsb-roberta-large", device=dev)
sp = set(tok.all_special_ids) | {MASK, 126348, 126081}
def dec(ids): return tok.decode([i for i in ids if i not in sp], skip_special_tokens=True).strip()
def sync(): torch.cuda.synchronize(dev)

def n_commit(conf):
    s = torch.sort(conf, descending=True)[0]; n = len(s)
    th = [1 - 1.0 / (k + 2) for k in range(n)]; th[0] = -1; top = 0
    for top in range(n):
        if s[top] < th[top]: break
    if top == 0 or top == n - 1: top += 1
    return top

@torch.no_grad()
def generate(ids, GEN, BLK, g=None, rec=None):
    """g=None: greedy (the demo answer). g=Generator: temperature-1 sample (Gumbel-max), as in cocoa.py."""
    Lp = ids.shape[1]; x = torch.full((1, Lp + GEN), MASK, dtype=torch.long, device=dev); x[:, :Lp] = ids
    drafts, blocks, clp, cst, step_t = [], [], [None] * GEN, [-1] * GEN, []
    for b in range(GEN // BLK):
        s, e = Lp + b * BLK, Lp + (b + 1) * BLK
        while (x[0, s:e] == MASK).any():
            t0 = time.perf_counter()
            lg = model(x).logits[0, Lp:]                      # bf16, gen region only
            lb = lg[b * BLK:(b + 1) * BLK].float()
            if g is None:
                lp = lb.log_softmax(-1); conf, x0 = lp.exp().max(-1)
            else:
                gum = -torch.log(-torch.log(torch.rand(lb.shape, generator=g, device=lb.device).clamp_min(1e-20)))
                x0 = (lb + gum).argmax(-1); lp = lb.log_softmax(-1); conf = lp.exp().gather(-1, x0[:, None])[:, 0]
            m = x[0, s:e] == MASK
            c = torch.where(m, conf, torch.full_like(conf, -1.0))
            sel = torch.topk(c, n_commit(conf[m])).indices
            if g is None:   # what D-CoCoA-L needs from the trajectory: argmax draft + commit log-prob
                am = lg.argmax(-1)
                if rec is not None:   # untimed recording run: full trace in the trace.py format
                    rec.append(dict(block=b, state=x[0, Lp:].tolist(), argmax=am.tolist(),
                                    conf=[round(v, 4) for v in lg.float().softmax(-1).max(-1)[0].tolist()],
                                    committed=[b * BLK + j for j in sel.tolist()]))
                drafts.append(torch.where(x[0, Lp:] == MASK, am, x[0, Lp:]).tolist()); blocks.append(b)
                for j in sel.tolist(): clp[b * BLK + j] = float(lp[j, x0[j]]); cst[b * BLK + j] = len(drafts) - 1
            x[0, s + sel] = x0[sel]
            sync(); step_t.append(time.perf_counter() - t0)
    generate.cst = cst
    return x[0, Lp:].tolist(), drafts, blocks, clp, step_t

def sims(pairs):
    out = [None] * len(pairs); todo = []
    for k, (a, b) in enumerate(pairs):
        if a == b: out[k] = 1.0
        elif not a or not b: out[k] = 0.0
        else: todo.append(k)
    if todo:
        for k, v in zip(todo, ce.predict([pairs[k] for k in todo], batch_size=64, show_progress_bar=False)): out[k] = float(v)
    return out

def d_cocoa_l(y, drafts, blocks, clp, BLK):
    ad = []
    for b in sorted(set(blocks)):
        fin = dec(y[b * BLK:(b + 1) * BLK])
        if not fin: continue
        ss = sims([(dec(drafts[t][b * BLK:(b + 1) * BLK]), fin) for t in range(len(drafts)) if blocks[t] == b])
        ad.append(float(np.mean([1 - v for v in ss])))
    valid = [k for k in range(len(y)) if y[k] not in sp and clp[k] is not None]
    cnll = float(np.mean([-clp[k] for k in valid])) if valid else 0.0
    return cnll * (float(np.mean(ad)) if ad else 0.0)

@torch.no_grad()
def mcnll(ids, y, g, bs):
    L = len(y)
    if L == 0: return 0.0
    seq = torch.cat([ids[0], torch.tensor(y, device=dev)])[None]; Lp = ids.shape[1]; tot = []
    for _ in range(NMC // bs):
        batch = seq.repeat(bs, 1); ls = []
        for k in range(bs):
            l = int(torch.randint(1, L + 1, (1,), generator=g, device=dev)); ls.append(l)
            batch[k, torch.randperm(L, generator=g, device=dev)[:l] + Lp] = MASK
        lp = model(batch).logits[:, Lp:].float().log_softmax(-1)
        for k in range(bs):
            msk = batch[k, Lp:] == MASK
            tot.append(float(-lp[k][msk].gather(-1, torch.tensor(y, device=dev)[msk][:, None]).sum()) * L / ls[k])
    return float(np.mean(tot))

def load(ds, i):
    f = H + ("traces.jsonl" if ds == "gsm8k" else f"traces_{ds}.jsonl")
    for line in open(f):
        r = json.loads(line)
        if r["i"] == i: return r

def timed(fn):
    sync(); t0 = time.perf_counter(); out = fn(); sync(); return out, time.perf_counter() - t0

res, traces = [], {}
for k, (ds, i) in enumerate(EX):
    r = load(ds, i)
    GEN, BLK = (256, 32) if ds == "gsm8k" else (20, 20)
    ids = tok.apply_chat_template([{"role": "user", "content": r["question"]}], add_generation_prompt=True,
                                  tokenize=True, return_tensors="pt", return_dict=True)["input_ids"].to(dev)
    if k == 0: generate(ids, GEN, BLK)   # warm-up (CUDA kernels, allocator)
    gens, dco, strm, coc = [], [], [], []
    for rep in range(REP):
        (y, drafts, blocks, clp, st), tg = timed(lambda: generate(ids, GEN, BLK))
        gens.append(dict(total=tg, steps=st))
        u, td = timed(lambda: d_cocoa_l(y, drafts, blocks, clp, BLK)); dco.append(td)
        a, b_ = dec(drafts[len(drafts) // 2]), dec(y)
        _, ts = timed(lambda: ce.predict([(a or ".", b_ or ".")], show_progress_bar=False)); strm.append(ts)
        g = torch.Generator(device=dev); g.manual_seed(1000 + i)
        L = next((q for q, v in enumerate(y) if v in sp), len(y))
        _, t1 = timed(lambda: mcnll(ids, y[:L], g, 2 if ds == "gsm8k" else 4))
        samples, t2 = timed(lambda: [dec(generate(ids, GEN, BLK, g)[0]) for _ in range(M)])
        _, t3 = timed(lambda: ce.predict([(b_, s) for s in samples], show_progress_bar=False))
        coc.append(dict(mcnll=t1, sampling=t2, ce=t3, total=t1 + t2 + t3))
    same = y == r["y"]
    steps = []; y2, _, _, clp2, _ = generate(ids, GEN, BLK, rec=steps); assert y2 == y
    text = tok.decode(y, skip_special_tokens=True).strip()
    tr = dict(i=i, question=r["question"], gold=r["gold"], text=text, y=y, steps=steps, commit_lp=clp2,
              commit_step=generate.cst, nfe=len(steps), secs=float(np.median([x["total"] for x in gens])))
    if ds != "gsm8k": tr["ds"] = ds
    traces.setdefault(ds, []).append(tr)
    med = lambda v: float(np.median(v))
    rec = dict(ds=ds, i=i, nfe=len(gens[0]["steps"]), same_output_as_demo=same, d_cocoa_l=u,
               gen_s=med([x["total"] for x in gens]), step_ms=1000 * med([np.median(x["steps"]) for x in gens]),
               step_times_ms=[round(1000 * v, 1) for v in gens[-1]["steps"]],
               dcocoa_s=med(dco), stream_ms=1000 * med(strm),
               cocoa_s=med([c["total"] for c in coc]), cocoa_parts_s={p: med([c[p] for c in coc]) for p in ("mcnll", "sampling", "ce")})
    res.append(rec)
    print(f"{ds:8s} {i:4d} nfe={rec['nfe']:3d} same={same} gen={rec['gen_s']:.2f}s step={rec['step_ms']:.0f}ms "
          f"D-CoCoA-L={rec['dcocoa_s']*1000:.0f}ms stream={rec['stream_ms']:.0f}ms/step CoCoA={rec['cocoa_s']:.1f}s "
          f"{ {p: round(v, 2) for p, v in rec['cocoa_parts_s'].items()} }", flush=True)
json.dump(dict(device=torch.cuda.get_device_name(dev), repeats=REP, res=res), open(H + "bench.json", "w"), indent=1)
for ds, T in traces.items():   # pred/correct are filled in by post-processing (same rules as trace*.py)
    with open(H + f"bench_traces_{ds}.jsonl", "w") as f:
        for t in T: f.write(json.dumps(t) + "\n")
print("saved", H + "bench.json")
