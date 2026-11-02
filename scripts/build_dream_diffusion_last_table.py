"""Build Dream-7B `all_estimators_llada_no_train_diffusion_last` results table.

Spans all 9 datasets (coqa, mmlu, gsm8k, samsum, truthfulqa, xsum, triviaqa,
wmt19_deen, wmt14_fren) with the 35-estimator ConfidenceAtCommit family.

Source pickles live on CIAI (`/home/artem.shelmanov/vlad/uncertainty_dllm/workdir/`)
under `ciai_<JOBID>_<DATASET>_simple_instruct_baselines/...`. Mirror them to
`PKL_DIR` below before running, named `<dataset>_diffusion_last.pkl`. To fetch:

    ssh ciai.mbzuai.ac.ae "cp /home/artem.shelmanov/vlad/uncertainty_dllm/workdir/ciai_<JOBID>_<DS>_simple_instruct_baselines/.../ue_manager_seed1 /tmp/<DS>_diffusion_last.pkl"
    scp ciai.mbzuai.ac.ae:/tmp/<DS>_diffusion_last.pkl /tmp/dream_diffusion_last_pkls/

Same shape as `build_dream_baselines_table.py` — wide table with one row per
estimator, columns = (dataset, metric) pairs of PRR_0.5_normalized values
plus a Mean / Mean-Rank summary. Output CSV at
`{PKL_DIR}/dream_diffusion_last_results.csv`.

Run:
    python scripts/build_dream_diffusion_last_table.py
"""
import os
import re
import string

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from lm_polygraph.ue_metrics.ue_metric import get_random_scores
from lm_polygraph.ue_metrics.pred_rej_area import PredictionRejectionArea


PKL_DIR = os.environ.get(
    "DREAM_DIFFUSION_LAST_PKL_DIR",
    "/tmp/dream_diffusion_last_pkls",
)
prr = PredictionRejectionArea(0.5)


# Per-dataset config: (pickle filename, list of generation metrics, special-handling flag).
# Generation metrics come from the dataset's gen_metrics — we report PRR/ECE
# vs each. Some datasets have degenerate Accuracy (mean ~0 for free-form
# tasks like xsum/samsum/truthfulqa) — those are dropped from the cross-
# dataset Mean below.
dataset_config = {
    "MMLU":       ("mmlu_diffusion_last.pkl",       ["Accuracy", "AlignScore"], False),
    "GSM8k":      ("gsm8k_diffusion_last.pkl",      ["Accuracy"],               True),
    "CoQA":       ("coqa_diffusion_last.pkl",       ["Accuracy", "AlignScore"], False),
    "TriviaQA":   ("triviaqa_diffusion_last.pkl",   ["Accuracy", "AlignScore"], False),
    "TruthfulQA": ("truthfulqa_diffusion_last.pkl", ["AlignScore"],             False),
    "SamSum":     ("samsum_diffusion_last.pkl",     ["AlignScore"],             False),
    "XSum":       ("xsum_diffusion_last.pkl",       ["AlignScore"],             False),
    "WMT14":      ("wmt14_diffusion_last.pkl",      ["Comet"],                  False),
    "WMT19":      ("wmt19_diffusion_last.pkl",      ["Comet"],                  False),
}


def get_table_short(path, metric_names, dataset, ue_metric="prr_0.5_normalized"):
    """Build a per-dataset PRR_0.5_normalized DataFrame: rows = methods, cols = metric_names."""
    man = torch.load(path, weights_only=False, map_location="cpu")
    df = {}
    methods = np.unique([x[1] for x in man["metrics"].keys() if "2side" not in x[1]])
    level = "sequence"

    # GSM8k has a known issue: post-hoc regex extraction recovers Accuracy=1
    # for some samples that the dataset's metric scored as 0. We mirror the
    # build_dream_baselines_table.py fix.
    if dataset == "GSM8k":
        greedy_texts = [x if len(x.strip()) > 0 else "-" for x in man["stats"]["greedy_texts"]]
        greedy_texts = np.array(greedy_texts)
        target_texts = np.array(man["stats"]["target_texts"])
        acc = np.array(man["gen_metrics"][("sequence", "Accuracy")])
        answers = []
        for text in greedy_texts[acc == 0]:
            answer_match = re.search(
                r"(?:The answer is|Answer[:]?)[^\d]*(\d+)",
                text,
                flags=re.IGNORECASE,
            )
            try:
                a = answer_match.group(1).strip().translate(
                    str.maketrans("", "", string.punctuation)
                )
                answers.append(int(a))
            except Exception:
                answers.append(-10**10)
        answers = np.array(answers)
        targets = []
        for text in target_texts[acc == 0]:
            output_ignore_regex = re.compile("(?s).*#### ")
            a = output_ignore_regex.sub("", text).translate(
                str.maketrans("", "", string.punctuation)
            )
            try:
                targets.append(int(a))
            except Exception:
                targets.append(-10**11)
        targets = np.array(targets)
        acc[acc == 0] = (answers == targets) * 1

    for method in methods:
        for metric in metric_names:
            key = (level, method, metric, ue_metric)
            scores = df.get(method, [])
            if dataset == "GSM8k" and metric == "Accuracy":
                # Recompute PRR_0.5_normalized against the regex-corrected acc.
                metrics_v = acc
                ue = np.array([
                    x if not isinstance(x, complex) else x.real
                    for x in man["estimations"][(level, method)]
                ])
                mean_val = prr(ue, metrics_v)
                oracle = prr(-metrics_v, metrics_v)
                random_v = get_random_scores(prr, metrics_v)
                scores.append((mean_val - random_v) / (oracle - random_v))
            else:
                scores.append(man["metrics"].get(key, np.nan))
            df[method] = scores

    for metric in metric_names:
        if dataset == "GSM8k" and metric == "Accuracy":
            print(f"  {metric}: {np.mean(acc):.4f}")
        else:
            try:
                print("  {} : {:.4f}".format(
                    metric,
                    np.mean(man["gen_metrics"][("sequence", metric)]),
                ))
            except KeyError:
                print(f"  {metric}: missing in pickle")

    df0 = pd.DataFrame(
        df,
        index=pd.MultiIndex.from_tuples([(dataset, m) for m in metric_names]),
    ).T
    df0.loc[:, (dataset, "Method")] = [x for x in df0.index]
    return df0.reset_index(drop=True)


def main():
    dfs = {}
    for ds_name, (pkl_name, metric_names, _flag) in dataset_config.items():
        p = os.path.join(PKL_DIR, pkl_name)
        if not os.path.exists(p):
            print(f"SKIP {ds_name}: missing {p}")
            continue
        print(f"Processing {ds_name} from {p}...")
        dfs[ds_name] = get_table_short(p, metric_names=metric_names, dataset=ds_name).reset_index(drop=True)

    if not dfs:
        raise RuntimeError(
            f"No pickles found in {PKL_DIR}. Mirror them from CIAI first; see this script's docstring."
        )

    # Pivot to wide format: one row per method, one column per (dataset, metric) pair.
    dfs_short = []
    methods = None
    for key in dfs.keys():
        df_ = dfs[key]
        if methods is None:
            methods = df_[df_.columns[-1]]
        df_ = df_.set_index(df_.columns[-1], drop=False).reindex(methods)
        if len(dfs_short):
            dfs_short.append(df_[df_.columns[:-1]])
        else:
            cols = list(df_.columns)
            dfs_short.append(df_[[cols[-1]] + cols[:-1]])
    dfs_short = pd.concat(dfs_short, axis=1).reset_index(drop=True)

    # Drop columns that are uninformative for the cross-dataset Mean:
    # - TruthfulQA's AlignScore is dominated by adversarial misconceptions
    #   (model agrees with the prompt's misleading premise → uninformative
    #   for UQ), per the colleague's earlier note in the no_train table.
    # - For the Mean we ALSO want to drop datasets where Accuracy is
    #   degenerate (mean ~0): SamSum, XSum, TruthfulQA's Accuracy is
    #   already absent from dataset_config so this is a no-op for them.
    drop_cols = [c for c in dfs_short.columns if c == ("TruthfulQA", "AlignScore")]
    metric_cols = [c for c in dfs_short.columns if c != dfs_short.columns[0]]

    dfs_short.loc[:, ("", "Mean")] = (
        dfs_short[metric_cols]
        .drop(columns=drop_cols, errors="ignore")
        .mean(axis=1)
    )
    dfs_short.loc[:, ("", "Mean Rank")] = (
        dfs_short[metric_cols]
        .drop(columns=drop_cols, errors="ignore")
        .apply(lambda x: rankdata(-x))
        .mean(axis=1)
        .round(2)
    )

    result = dfs_short.sort_values(("", "Mean"), ascending=False)

    out_path = os.path.join(PKL_DIR, "dream_diffusion_last_results.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved CSV to {out_path}; shape={result.shape}")
    print("\nTop 30 by Mean PRR_0.5_normalized:")
    print(result.head(30).to_string())


if __name__ == "__main__":
    main()
