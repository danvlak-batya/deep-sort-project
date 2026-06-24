"""Run original DeepSORT baseline using det/ files on Videos-CV layout."""
from __future__ import print_function

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deep_sort_app
from eval.mot_metrics import EVAL_SEQUENCES, evaluate_hota, save_metrics_json
from tools.generate_detections import create_box_encoder
from utils.mot_paths import find_sequence_dir, get_det_file, list_image_filenames


def build_npy_from_det(seq_path, encoder, npy_path):
    """Build DeepSORT .npy from det.txt + frames (mars-small128 features)."""
    det_path = get_det_file(seq_path)
    if not os.path.exists(det_path):
        raise FileNotFoundError("No det file: %s" % det_path)

    detections_in = np.loadtxt(det_path, delimiter=",")
    if detections_in.ndim == 1:
        detections_in = detections_in.reshape(1, -1)

    image_filenames = list_image_filenames(seq_path)
    frame_indices = detections_in[:, 0].astype(np.int64)
    min_frame = int(frame_indices.min())
    max_frame = int(frame_indices.max())

    detections_out = []
    for frame_idx in range(min_frame, max_frame + 1):
        if frame_idx % 50 == 0:
            print("  frame %d / %d" % (frame_idx, max_frame))
        mask = frame_indices == frame_idx
        rows = detections_in[mask]
        if len(rows) == 0:
            continue
        if frame_idx not in image_filenames:
            print("  WARNING: no image for frame %d" % frame_idx)
            continue
        bgr_image = cv2.imread(image_filenames[frame_idx], cv2.IMREAD_COLOR)
        if bgr_image is None:
            continue
        features = encoder(bgr_image, rows[:, 2:6].copy())
        for row, feature in zip(rows, features):
            detections_out.append(np.r_[row, feature])

    if not detections_out:
        raise RuntimeError("No detections built for %s" % seq_path)

    os.makedirs(os.path.dirname(npy_path) or ".", exist_ok=True)
    np.save(npy_path, np.asarray(detections_out), allow_pickle=False)
    print("Saved:", npy_path, "rows:", len(detections_out))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline DeepSORT from course det/ files (no prebuilt .npy)")
    parser.add_argument("--mot_root", required=True)
    parser.add_argument("--model", default="resources/networks/mars-small128.pb")
    parser.add_argument("--npy_dir", default="resources/detections")
    parser.add_argument("--output_dir", default="results/baseline")
    parser.add_argument("--min_confidence", type=float, default=0.3)
    parser.add_argument("--max_cosine_distance", type=float, default=0.2)
    parser.add_argument("--nn_budget", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.model):
        raise FileNotFoundError(
            "ReID model not found: %s\n"
            "Download resources/networks/ from original DeepSORT Drive." % args.model)

    os.makedirs(args.npy_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading mars-small128 encoder...")
    encoder = create_box_encoder(args.model, batch_size=32)

    prepared = []
    for seq in EVAL_SEQUENCES:
        try:
            seq_path = find_sequence_dir(args.mot_root, seq)
        except FileNotFoundError:
            print("Skip %s: folder not found" % seq)
            continue

        npy_path = os.path.join(args.npy_dir, "%s.npy" % seq)
        if not os.path.exists(npy_path):
            print("Building %s from det/ ..." % seq)
            build_npy_from_det(seq_path, encoder, npy_path)

        out_file = os.path.join(args.output_dir, "%s.txt" % seq)
        print("Baseline tracking:", seq)
        deep_sort_app.run(
            seq_path, npy_path, out_file,
            args.min_confidence, 1.0, 0,
            args.max_cosine_distance, args.nn_budget, display=False)
        prepared.append(seq)

    if not prepared:
        raise RuntimeError("No sequences processed.")

    metrics = evaluate_hota(args.mot_root, args.output_dir, sequences=prepared)
    hota_path = os.path.join(args.output_dir, "hota.json")
    save_metrics_json(metrics, hota_path)
    print("Saved:", hota_path)
    print("Mean baseline HOTA:", metrics["mean_hota"])


if __name__ == "__main__":
    main()
