"""Generate video overlays with tracking results."""
from __future__ import print_function

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.mot_metrics import _find_sequence_path
from utils.mot_paths import list_image_filenames


def generate_overlay(sequence_path, result_file, output_video, fps=25):
    image_filenames = list_image_filenames(sequence_path)
    results = np.loadtxt(result_file, delimiter=",")
    if results.ndim == 1:
        results = results.reshape(1, -1)

    frames = sorted(image_filenames.keys())
    first = cv2.imread(next(iter(image_filenames.values())))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h))

    np.random.seed(42)
    colors = {}

    for frame_idx in frames:
        image_path = image_filenames.get(frame_idx)
        if image_path is None:
            continue
        image = cv2.imread(image_path)
        if image is None:
            continue
        mask = results[:, 0].astype(int) == frame_idx
        rows = results[mask]
        for row in rows:
            track_id = int(row[1])
            x, y, bw, bh = row[2:6]
            if track_id not in colors:
                colors[track_id] = tuple(int(c) for c in np.random.randint(64, 255, 3))
            color = colors[track_id]
            cv2.rectangle(image, (int(x), int(y)), (int(x + bw), int(y + bh)), color, 2)
            cv2.putText(
                image, "ID %d" % track_id, (int(x), max(0, int(y) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        writer.write(image)
    writer.release()
    print("Saved overlay:", output_video)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate tracking overlay videos")
    parser.add_argument("--mot_dir", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", default="results/overlays")
    parser.add_argument("--sequence", help="Single sequence name")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    sequences = [args.sequence] if args.sequence else [
        f[:-4] for f in os.listdir(args.results_dir) if f.endswith(".txt")]
    for seq in sequences:
        seq_path = _find_sequence_path(args.mot_dir, seq)
        result_file = os.path.join(args.results_dir, "%s.txt" % seq)
        if not os.path.exists(result_file):
            print("Skip %s: no results" % seq)
            continue
        out_video = os.path.join(args.output_dir, "%s_overlay.mp4" % seq)
        generate_overlay(seq_path, result_file, out_video)


if __name__ == "__main__":
    main()
