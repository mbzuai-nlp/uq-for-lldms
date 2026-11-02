"""Build Dream-7B with-train BASELINE results table for WMT (setup B).

Setup B scope: WMT translation tasks evaluated with the full ~150-estimator
config `baseline_estimators_llada` (no_train baselines + the 6 trained
methods together — TAD, LookBackLens, SATRMD, SAPLMA, MIND, MSP), with
the colleague's translation-specific overrides:
    target_metric=Comet
    metric_thr=0.85    (×3, on estimators 50/51/52)

Currently in scope: wmt14_fren simple_instruct, wmt19_deen simple_instruct,
wmt14_fren OLD prompt, wmt19_deen OLD prompt.

Source pickles: each run was launched on either CIAI (slurm) or local mbz
nodes. Pickle naming convention used in this repo:

    /tmp/dream_with_train_pkls/{wmt14_fren_simple,wmt19_deen_simple,
                                wmt14_fren_old,wmt19_deen_old}.pkl

Output: `dream_with_train_results.csv` in `PKL_DIR` — wide table, rows =
methods (sorted by Mean Comet PRR_0.5_normalized desc), columns =
(dataset, Comet) PRR_0.5_normalized values plus Method, Mean, Mean Rank.

The builder skips datasets whose pickles are missing, so it can be run
incrementally as runs land — currently only wmt14_fren simple_instruct
is complete.

Run:
    python scripts/build_dream_with_train_table.py
"""
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata


PKL_DIR = os.environ.get(
    "DREAM_WITH_TRAIN_PKL_DIR",
    "/tmp/dream_with_train_pkls",
)


# Per-dataset config: (pickle filename, list of generation metrics).
# Translation tasks use Comet; metric_thr=0.85 was applied during training
# of the 3 trained classifier-style estimators.
dataset_config = {
    "WMT14_fren_simple": ("wmt14_fren_simple.pkl", ["Comet"]),
    "WMT19_deen_simple": ("wmt19_deen_simple.pkl", ["Comet"]),
    "WMT14_fren_old":    ("wmt14_fren_old.pkl",    ["Comet"]),
    "WMT19_deen_old":    ("wmt19_deen_old.pkl",    ["Comet"]),
}


def get_table_short(path, metric_names, dataset, ue_metric="prr_0.5_normalized"):
    """Per-dataset PRR_0.5_normalized DataFrame: rows = methods, cols = metric_names."""
    man = torch.load(path, weights_only=False, map_location="cpu")
    df = {}
    methods = np.unique([x[1] for x in man["metrics"].keys() if "2side" not in x[1]])
    level = "sequence"

    for method in methods:
        for metric in metric_names:
            key = (level, method, metric, ue_metric)
            scores = df.get(method, [])
            scores.append(man["metrics"].get(key, np.nan))
            df[method] = scores

    for metric in metric_names:
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
    for ds_name, (pkl_name, metric_names) in dataset_config.items():
        p = os.path.join(PKL_DIR, pkl_name)
        if not os.path.exists(p):
            print(f"SKIP {ds_name}: missing {p}")
            continue
        print(f"Processing {ds_name} from {p}...")
        dfs[ds_name] = get_table_short(p, metric_names=metric_names, dataset=ds_name).reset_index(drop=True)

    if not dfs:
        raise RuntimeError(
            f"No pickles found in {PKL_DIR}. Mirror them from the workdir(s) first."
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

    metric_cols = [c for c in dfs_short.columns if c != dfs_short.columns[0]]

    dfs_short.loc[:, ("", "Mean")] = dfs_short[metric_cols].mean(axis=1)
    dfs_short.loc[:, ("", "Mean Rank")] = (
        dfs_short[metric_cols]
        .apply(lambda x: rankdata(-x))
        .mean(axis=1)
        .round(2)
    )

    result = dfs_short.sort_values(("", "Mean"), ascending=False)

    out_path = os.path.join(PKL_DIR, "dream_with_train_results.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved CSV to {out_path}; shape={result.shape}")
    print("\nTop 30 by Mean Comet PRR_0.5_normalized:")
    print(result.head(30).to_string())


if __name__ == "__main__":
    main()
