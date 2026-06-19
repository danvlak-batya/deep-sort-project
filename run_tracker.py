"""CLI entry point for modern DeepSORT pipeline."""
from __future__ import print_function

import argparse
import json
import os

from pipeline.config import load_config
from pipeline.run_tracker import run_mot_directory, run_sequence


def parse_args():
    parser = argparse.ArgumentParser(description="Modern DeepSORT tracker")
    parser.add_argument("--sequence_dir", help="Single MOT sequence directory")
    parser.add_argument("--mot_dir", help="MOT directory with multiple sequences")
    parser.add_argument("--output_file", help="Output file for single sequence")
    parser.add_argument("--output_dir", default="results/modern", help="Output directory")
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    parser.add_argument("--detector", help="Override detector name")
    parser.add_argument("--reid", help="Override ReID model name")
    parser.add_argument("--display", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    video_name = None
    if args.sequence_dir:
        video_name = os.path.basename(args.sequence_dir.rstrip("/\\"))
    config = load_config(args.config, video_name=video_name)
    if args.detector:
        config["detector"] = args.detector
    if args.reid:
        config["reid"] = args.reid

    if args.sequence_dir:
        output = args.output_file or os.path.join(
            args.output_dir, "%s.txt" % os.path.basename(args.sequence_dir.rstrip("/\\")))
        stats = run_sequence(
            args.sequence_dir, config, output_file=output,
            display=args.display, config_path=args.config)
        print(json.dumps(stats, indent=2, default=str))
    elif args.mot_dir:
        stats = run_mot_directory(args.mot_dir, config, args.output_dir, config_path=args.config)
        summary = {s["sequence_name"]: {"fps": round(s["fps"], 2), "frames": s["num_frames"]} for s in stats}
        print(json.dumps(summary, indent=2))
    else:
        raise SystemExit("Provide --sequence_dir or --mot_dir")


if __name__ == "__main__":
    main()
