"""Run original DeepSORT baseline on pre-generated detections."""
from __future__ import print_function

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deep_sort_app
from eval.mot_metrics import EVAL_SEQUENCES, _find_sequence_path, evaluate_hota, save_metrics_json


def parse_args():
    parser = argparse.ArgumentParser(description="Original DeepSORT baseline")
    parser.add_argument("--mot_root", required=True)
    parser.add_argument("--detections_dir", required=True,
                        help="Folder with <sequence>.npy detection files")
    parser.add_argument("--output_dir", default="results/baseline")
    parser.add_argument("--min_confidence", type=float, default=0.3)
    parser.add_argument("--max_cosine_distance", type=float, default=0.2)
    parser.add_argument("--nn_budget", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for seq in EVAL_SEQUENCES:
        det_file = os.path.join(args.detections_dir, "%s.npy" % seq)
        if not os.path.exists(det_file):
            print("Skip %s: no detections" % seq)
            continue
        seq_path = _find_sequence_path(args.mot_root, seq)
        out_file = os.path.join(args.output_dir, "%s.txt" % seq)
        print("Baseline:", seq)
        deep_sort_app.run(
            seq_path, det_file, out_file,
            args.min_confidence, 1.0, 0,
            args.max_cosine_distance, args.nn_budget, display=False)

    metrics = evaluate_hota(args.mot_root, args.output_dir, sequences=EVAL_SEQUENCES,
                            tracker_name="deep_sort")
    save_metrics_json(metrics, os.path.join(args.output_dir, "hota.json"))
    print("Mean HOTA:", metrics["mean_hota"])


if __name__ == "__main__":
    main()
