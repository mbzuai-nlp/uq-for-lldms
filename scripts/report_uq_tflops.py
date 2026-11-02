#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


BASE_COMPONENT = "DiffusionGreedyProbsCalculator"

METHOD_COMPONENTS = {
    "KLE": {
        "stat_calculators": [
            "SamplingGenerationCalculator",
            "SemanticMatrixCalculator",
        ],
        "estimators": ["KernelLanguageEntropy"],
    },
    "CoCoA-MCNLL": {
        "stat_calculators": [
            "SamplingGenerationCalculator",
            "GreedyCrossEncoderSimilarityMatrixCalculator",
            "MCMaskedLLCalculator",
        ],
        "estimators": ["CocoaMCMaskedNLL"],
    },
    "CommitNLL": {
        "stat_calculators": [],
        "estimators": ["ConfidenceAtCommit_avg"],
    },
    "D-CoCoA-L": {
        "stat_calculators": [
            "GreedyCrossEncoderSimilarityMatrixBlockTraceCalculator",
        ],
        "estimators": [
            "CocoaConfidenceAtCommitCrossEncoderBlockAvgTrace",
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Report per-example TFLOPs for the four profiled UQ methods."
    )
    parser.add_argument("manager_path", type=Path)
    return parser.parse_args()


def component_flops(profile, group, name):
    return float(sum(profile.get(group, {}).get(name, [])))


def main():
    args = parse_args()
    run = torch.load(args.manager_path, map_location="cpu", weights_only=False)
    profile = run.get("flop_stats")
    if not profile:
        raise SystemExit(
            "No FLOP profile found. Run polygraph_eval with +log_flops=true."
        )

    n_examples = len(run.get("stats", {}).get("greedy_texts", []))
    if n_examples == 0:
        raise SystemExit("Cannot determine the number of evaluated examples.")

    base_flops = component_flops(
        profile, "stat_calculators", BASE_COMPONENT
    )
    if base_flops == 0:
        available = sorted(profile.get("stat_calculators", {}))
        raise SystemExit(
            f"Missing {BASE_COMPONENT}. Available calculators: {available}"
        )

    scale = n_examples * 1e12
    print(f"Examples: {n_examples}")
    print(
        f"{'Method':<16} {'Base':>14} {'Overhead':>14} {'Total':>14}"
    )
    print(
        f"{'':<16} {'TFLOP/example':>14} {'TFLOP/example':>14} "
        f"{'TFLOP/example':>14}"
    )

    for method, groups in METHOD_COMPONENTS.items():
        overhead_flops = 0.0
        missing = []
        for group, names in groups.items():
            for name in names:
                values = profile.get(group, {}).get(name)
                if values is None:
                    missing.append(f"{group}:{name}")
                    continue
                overhead_flops += float(sum(values))

        if missing:
            raise SystemExit(
                f"{method}: missing profile components: {', '.join(missing)}"
            )

        print(
            f"{method:<16} "
            f"{base_flops / scale:14.6f} "
            f"{overhead_flops / scale:14.6f} "
            f"{(base_flops + overhead_flops) / scale:14.6f}"
        )

    print()
    print(
        "NumPy/SciPy estimator post-processing is below the displayed "
        "precision; neural-model and Torch estimator FLOPs are counted."
    )


if __name__ == "__main__":
    main()
