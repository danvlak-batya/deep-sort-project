"""Grid search over tracker hyperparameters."""
from __future__ import print_function

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.mot_metrics import EVAL_SEQUENCES, evaluate_hota
from pipeline.config import load_config
from pipeline.run_tracker import run_mot_directory


def parse_args():
    parser = argparse.ArgumentParser(description="Tracker parameter grid search")
    parser.add_argument("--mot_dir", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_dir", default="results/tuning")
    parser.add_argument("--sequences", nargs="*", default=EVAL_SEQUENCES[:2],
                        help="Subset of sequences for faster search")
    parser.add_argument("--detector", default="yolov8n")
    parser.add_argument("--reid", default="osnet_x0_25")
    return parser.parse_args()


def main():
    args = parse_args()
    base = load_config(args.config)
    base["detector"] = args.detector
    base["reid"] = args.reid

    grid = {
        "max_cosine_distance": [0.15, 0.2, 0.25],
        "nms_max_overlap": [0.6, 0.7, 0.8],
        "min_confidence": [0.25, 0.3, 0.35],
        "nn_budget": [50, 100],
    }

    keys = list(grid.keys())
    results = []
    for values in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, values))
        trial_cfg = dict(base)
        trial_cfg.update(params)
        trial_name = "_".join("%s=%s" % (k, v) for k, v in params.items())
        trial_dir = os.path.join(args.output_dir, trial_name)
        print("Trial:", trial_name)
        run_mot_directory(args.mot_dir, trial_cfg, trial_dir, config_path=args.config)
        metrics = evaluate_hota(args.mot_dir, trial_dir, sequences=args.sequences)
        entry = {"params": params, "mean_hota": metrics["mean_hota"], "per_sequence": metrics["per_sequence"]}
        results.append(entry)
        print("  mean HOTA:", metrics["mean_hota"])

    results.sort(key=lambda x: x["mean_hota"], reverse=True)
    out_path = os.path.join(args.output_dir, "grid_search_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Best params:", results[0]["params"] if results else "none")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
