# How the landing-page demo was made

> **Current demo:** all 9 examples, scores and timings come from the A100 re-recording described below (section on timings, `a100/`). The earlier sections document how the first version was built; the scripts are the same.

Both animations on the project page are **real LLaDA-1.5 generations** on GSM8K (test split,
`LM-Polygraph/gsm8k`, `simple_instruct` prompt), not mock-ups.

| file | what it does |
|---|---|
| `trace.py` | Greedy LLaDA-1.5 decoding with the paper setup (256 new tokens, block 32, dynamic confidence-based unmasking as in `llada/generate.py`, `factor=1`). Stores the state, the argmax guess and its confidence for every position at every step. |
| `score.py` | Final UQ scores for all recorded generations: CommitNLL, AD-Block, AD-Full and **D-CoCoA-L = CommitNLL × AD-Block** (cross-encoder `cross-encoder/stsb-roberta-large`, D = 1 − similarity). |
| `viz.py` | Per-frame data for selected examples, including the running score. |
| `frame.html`, `make_gif.py` | Renders each denoising step with Playwright and assembles the GIF / MP4. |
| `viz_demo.json` | Frame data of the two examples shown on the page (GSM8K test ids 279 and 182). |

**Running score.** At step *t* the panel shows D-CoCoA-L computed on the trajectory prefix:
the dissimilarity term is AD-Block where the still-running block uses its current draft as the
provisional final text, and the NLL term is CommitNLL where not-yet-committed tokens use the
current argmax confidence. At the last step both terms coincide with the paper's definitions, so
the last value is the final D-CoCoA-L score.

**Threshold.** The flag line is the 80th percentile of final D-CoCoA-L over 109 recorded GSM8K
generations (reject the 20% most uncertain). Example 279 ends at the 18th percentile (accepted,
correct answer 6); example 182 ends at the 96th percentile (flagged, wrong answer 9 vs. 23).

Run from the repo root with `PYTHONPATH=.` (`trace.py` imports `llada.modeling_llada`). The `max_memory` split in `trace.py` was set for a shared 2-GPU box; change it for your hardware.

```bash
python trace.py 300 0            # -> ~/dllm_demo/traces.jsonl
python score.py cuda:0           # -> ~/dllm_demo/scores.json
CE_DEV=cuda:0 python viz.py 279,182
python make_gif.py viz_demo.json 279 out/demo_correction --thr 0.0182 --pct 18 --scale 2
python make_gif.py viz_demo.json 182 out/demo_flagged    --thr 0.0182 --pct 4  --scale 2
```

## QA examples (TriviaQA, CoQA)

`trace_qa.py DS N START` records LLaDA-1.5 on `LM-Polygraph/triviaqa` / `LM-polygraph/coqa`
(`simple_instruct`, 20 new tokens as in the paper; for simplicity decoded as one 20-token block, while the paper uses block length 5). `viz_qa.py DS [ids]` computes
final D-CoCoA-L for every recorded example plus frame data for the chosen ids; correctness is a
normalised exact/contains match against the reference aliases (a rough label, not AlignScore).
Recorded: TriviaQA 250 (test 0-249), CoQA 150 (test 0-149). Frame data of the shown examples: `viz_qa_demo.json`.

## Score normalisation (all animations)

The displayed score is min-max normalised to [0, 1] per dataset in log space:
`clip((log u - log lo) / (log hi - log lo), 0, 1)`, with `lo` / `hi` = the 1st / 99th percentile of the
dataset's final D-CoCoA-L scores (GSM8K 109, TriviaQA 250, CoQA 150 generations). Running values above
`hi` are clipped to 1. The flag threshold (80th percentile of final scores) is mapped the same way.
Exact bounds: `r3.sh` / `rc.sh`. Shown examples (3 per dataset: hallucination, confident correct answer,
self-correction): GSM8K 1046/599/279, TriviaQA 92/20/99, CoQA 50/146/12 (current, A100-recorded: `a100/viz_*.json`; the earlier set 182/235/149 recorded on the .42 server is in `viz_all_demo.json`).

## "vs. standard CoCoA"

`cocoa.py DS N ids` computes standard CoCoA-MCNLL for recorded generations:
MCNLL of the greedy answer (16 Monte Carlo masking passes, Eq. 1 of the paper) × mean cross-encoder
dissimilarity to 5 temperature-1 samples. It was run on the first 20–45 recorded examples of each dataset
plus the demo examples (GSM8K 25, TriviaQA 49, CoQA 39); `cnorm.py` gives the same log min–max bounds
(1st/99th pct) and 80th-percentile threshold that the animations use for CoCoA's 0–1 score.

Everything in the animations was recorded and timed on ONE GPU: an NVIDIA A100-SXM4-40GB on the CIAI cluster
(LLaDA-1.5 in bf16, no CPU offload, one Slurm job with a whole GPU). `rec.py` re-recorded all traces
(GSM8K 109, TriviaQA 250, CoQA 150) on that GPU; `score.py` / `viz_qa.py` recomputed every D-CoCoA-L score,
the normalisation bounds and thresholds; `cocoa.py` recomputed standard CoCoA on the same subsets; `bench.py`
timed the 9 demo examples (median of 3 runs; its re-generated outputs are identical to the recorded traces).
Results are in `a100/` (bench.json, scores*.json, cocoa_*.jsonl, viz_*.json, cnorm.log).
- real-time mode: every frame lasts its own measured wall-clock = that step's denoising time (bench.json,
  `step_times_ms`) + that step's live D-CoCoA update (`stream.py` -> a100/stream.json, median of 3). `stream.py`
  runs exactly the running-score maths of viz.py / viz_qa.py, incrementally: cross-encoder scores that cannot change
  are cached, the rest go in one batch. Denoising: GSM8K ~104 ms, TriviaQA ~34 ms, CoQA 40-66 ms per step;
  live update: 0.1-41 ms per step (mean 0.1-19 ms; <1 ms whenever the draft did not change).
- "ready after generation": D-CoCoA-L = CommitNLL + AD-Block on the stored drafts: <1-23 ms (QA), 106-264 ms (GSM8K);
- standard CoCoA-MCNLL: 16 masked passes + 5 temperature-1 samples + cross-encoder: 1.1-4.1 s (QA), 22.6-42.4 s (GSM8K).
Render commands: `r6.sh`, `q7a.sh`, `q7b.sh` (per-step times in `steptimes/`).
Waits longer than 15 s are fast-forwarded (shown with ⏩ and the real number of seconds).
