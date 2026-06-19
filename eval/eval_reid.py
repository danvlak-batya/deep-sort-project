"""Evaluate ReID quality using ground-truth bounding boxes."""
from __future__ import print_function

import argparse
import json
import os
import sys

import cv2
import numpy as np
from sklearn.metrics import calinski_harabasz_score, fowlkes_mallows_score, silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.mot_metrics import EVAL_SEQUENCES, _find_sequence_path
from reid import create_reid, list_reid_models


def load_gt_crops(sequence_path, max_frames=None):
    img_dir = os.path.join(sequence_path, "img1")
    gt_path = os.path.join(sequence_path, "gt", "gt.txt")
    gt = np.loadtxt(gt_path, delimiter=",")
    if gt.ndim == 1:
        gt = gt.reshape(1, -1)

    frames = sorted(set(gt[:, 0].astype(int)))
    if max_frames:
        frames = frames[:max_frames]

    crops_meta = []
    for frame_idx in frames:
        image_path = os.path.join(img_dir, "%06d.jpg" % frame_idx)
        image = cv2.imread(image_path)
        if image is None:
            image = cv2.imread(os.path.join(img_dir, "%06d.png" % frame_idx))
        mask = gt[:, 0].astype(int) == frame_idx
        for row in gt[mask]:
            if len(row) > 7 and int(row[7]) not in (1, 2, 7):
                continue
            crops_meta.append({
                "image": image,
                "box": row[2:6],
                "track_id": int(row[1]),
                "frame": int(frame_idx),
            })
    return crops_meta


def evaluate_reid_on_sequence(encoder, sequence_path, max_frames=200):
    meta = load_gt_crops(sequence_path, max_frames=max_frames)
    if not meta:
        return {}

    features = []
    labels = []
    by_frame = {}
    for item in meta:
        by_frame.setdefault(item["frame"], {"image": item["image"], "boxes": [], "ids": []})
        by_frame[item["frame"]]["boxes"].append(item["box"])
        by_frame[item["frame"]]["ids"].append(item["track_id"])

    for frame_idx, data in sorted(by_frame.items()):
        boxes = np.array(data["boxes"])
        emb = encoder.encode(data["image"], boxes)
        features.append(emb)
        labels.extend(data["ids"])

    features = np.vstack(features)
    labels = np.array(labels)
    unique_labels = len(set(labels.tolist()))
    if unique_labels < 2 or len(features) < unique_labels + 1:
        return {"fowlkes_mallows": 0.0, "silhouette": 0.0, "calinski_harabasz": 0.0}

    # Cluster embeddings with track IDs as ground-truth clusters for extrinsic metrics
    fmi = fowlkes_mallows_score(labels, labels)
    try:
        sil = silhouette_score(features, labels, metric="cosine")
    except Exception:
        sil = 0.0
    try:
        ch = calinski_harabasz_score(features, labels)
    except Exception:
        ch = 0.0

    # Nearest-neighbor identity accuracy within same frame (association proxy)
    nn_correct = 0
    nn_total = 0
    for frame_idx, data in sorted(by_frame.items()):
        boxes = np.array(data["boxes"])
        ids = np.array(data["ids"])
        if len(boxes) < 2:
            continue
        emb = encoder.encode(data["image"], boxes)
        from sklearn.metrics.pairwise import cosine_distances
        dist = cosine_distances(emb)
        np.fill_diagonal(dist, np.inf)
        for i in range(len(ids)):
            j = np.argmin(dist[i])
            nn_total += 1
            if ids[i] == ids[j]:
                nn_correct += 1
    nn_acc = nn_correct / nn_total if nn_total else 0.0

    return {
        "fowlkes_mallows": float(fmi),
        "silhouette": float(sil),
        "calinski_harabasz": float(ch),
        "nn_same_id_accuracy": float(nn_acc),
        "num_samples": int(len(features)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="ReID evaluation with GT boxes")
    parser.add_argument("--mot_dir", required=True)
    parser.add_argument("--reid", default="osnet_x0_25", choices=list_reid_models())
    parser.add_argument("--sequences", nargs="*", default=EVAL_SEQUENCES)
    parser.add_argument("--max_frames", type=int, default=200)
    parser.add_argument("--output", default="results/reid_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    encoder = create_reid(args.reid)
    results = {"reid": args.reid, "per_sequence": {}}
    for seq in args.sequences:
        seq_path = _find_sequence_path(args.mot_dir, seq)
        metrics = evaluate_reid_on_sequence(encoder, seq_path, max_frames=args.max_frames)
        results["per_sequence"][seq] = metrics
        print("%s: %s" % (seq, metrics))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
