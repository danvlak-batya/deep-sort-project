"""Run tracker on all benchmark sequences (MOT15 + MOT16)."""
from __future__ import print_function

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.mot_metrics import EVAL_SEQUENCES
from utils.mot_paths import find_sequence_dir
from pipeline.config import load_config
from pipeline.run_tracker import run_sequence


def parse_args():
    parser = argparse.ArgumentParser(description="Run all benchmark sequences")
    parser.add_argument("--mot_root", required=True, help="Root folder containing MOT15/MOT16")
    parser.add_argument("--output_dir", default="results/modern")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--detector", default="yolov8n")
    parser.add_argument("--reid", default="osnet_x0_25")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    all_stats = []
    for seq in EVAL_SEQUENCES:
        try:
            seq_path = find_sequence_dir(args.mot_root, seq)
        except FileNotFoundError:
            print("Skip %s: not found" % seq)
            continue
        cfg = load_config(args.config, video_name=seq)
        cfg["detector"] = args.detector
        cfg["reid"] = args.reid
        out_file = os.path.join(args.output_dir, "%s.txt" % seq)
        print("Running %s ..." % seq)
        stats = run_sequence(seq_path, cfg, output_file=out_file, config_path=args.config)
        all_stats.append(stats)
        print("  FPS: %.2f" % stats["fps"])

    summary_path = os.path.join(args.output_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
