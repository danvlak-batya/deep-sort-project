"""Run full MOT evaluation pipeline and compute HOTA."""
from __future__ import print_function

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.mot_metrics import EVAL_SEQUENCES, evaluate_hota, save_metrics_json
from pipeline.config import load_config
from pipeline.run_tracker import run_mot_directory


def parse_args():
    parser = argparse.ArgumentParser(description="Run tracker and evaluate HOTA")
    parser.add_argument("--mot_dir", required=True, help="MOT dataset root")
    parser.add_argument("--output_dir", default="results/modern")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--detector")
    parser.add_argument("--reid")
    parser.add_argument("--sequences", nargs="*", default=EVAL_SEQUENCES)
    parser.add_argument("--metrics_out", default="results/metrics_hota.json")
    parser.add_argument("--skip_tracking", action="store_true",
                        help="Only evaluate existing result files")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {}
    if args.detector:
        overrides["detector"] = args.detector
    if args.reid:
        overrides["reid"] = args.reid

    if not args.skip_tracking:
        run_mot_directory(args.mot_dir, overrides, args.output_dir, config_path=args.config)

    metrics = evaluate_hota(args.mot_dir, args.output_dir, sequences=args.sequences)
    save_metrics_json(metrics, args.metrics_out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
