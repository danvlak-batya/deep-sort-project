"""Evaluate detector quality: Precision, Recall, F1 vs MOT ground truth."""
from __future__ import print_function

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors import create_detector, list_detectors
from eval.mot_metrics import EVAL_SEQUENCES, _find_sequence_path
from utils.image import iou_matrix


IOU_THRESHOLD = 0.5


def load_gt_boxes(gt_path, frame_idx):
    if not os.path.exists(gt_path):
        return []
    gt = np.loadtxt(gt_path, delimiter=",")
    if gt.ndim == 1:
        gt = gt.reshape(1, -1)
    mask = gt[:, 0].astype(int) == frame_idx
    rows = gt[mask]
    boxes = []
    for row in rows:
        if len(row) > 7 and int(row[7]) not in (1, 2, 7):
            continue
        boxes.append(row[2:6])
    return boxes


def match_detections(pred_boxes, gt_boxes, iou_thresh=IOU_THRESHOLD):
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 0, 0, 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0

    ious = iou_matrix(pred_boxes, gt_boxes)
    matched_gt = set()
    tp = 0
    for i in range(len(pred_boxes)):
        best_j = np.argmax(ious[i])
        if ious[i, best_j] >= iou_thresh and best_j not in matched_gt:
            tp += 1
            matched_gt.add(best_j)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def evaluate_detector_on_sequence(detector, sequence_path):
    img_dir = os.path.join(sequence_path, "img1")
    gt_path = os.path.join(sequence_path, "gt", "gt.txt")
    frames = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(img_dir))
    tp = fp = fn = 0
    for frame_idx in frames:
        image = cv2.imread(os.path.join(img_dir, "%06d.jpg" % frame_idx))
        if image is None:
            image = cv2.imread(os.path.join(img_dir, "%06d.png" % frame_idx))
        dets = detector.detect(image)
        pred_boxes = [d[0] for d in dets]
        gt_boxes = load_gt_boxes(gt_path, frame_idx)
        t, f_p, f_n = match_detections(pred_boxes, gt_boxes)
        tp += t
        fp += f_p
        fn += f_n

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def parse_args():
    parser = argparse.ArgumentParser(description="Detector F1 evaluation")
    parser.add_argument("--mot_dir", required=True)
    parser.add_argument("--detector", default="yolov8n", choices=list_detectors())
    parser.add_argument("--sequences", nargs="*", default=EVAL_SEQUENCES)
    parser.add_argument("--output", default="results/detector_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    detector = create_detector(args.detector)
    results = {"detector": args.detector, "per_sequence": {}, "mean_f1": 0.0}
    f1_vals = []
    for seq in args.sequences:
        seq_path = _find_sequence_path(args.mot_dir, seq)
        metrics = evaluate_detector_on_sequence(detector, seq_path)
        results["per_sequence"][seq] = metrics
        f1_vals.append(metrics["f1"])
        print("%s: P=%.3f R=%.3f F1=%.3f" % (
            seq, metrics["precision"], metrics["recall"], metrics["f1"]))
    results["mean_f1"] = float(np.mean(f1_vals)) if f1_vals else 0.0
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Mean F1: %.3f" % results["mean_f1"])


if __name__ == "__main__":
    main()
