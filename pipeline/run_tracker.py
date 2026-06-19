"""Unified DeepSORT tracking pipeline with modern detectors and ReID."""
from __future__ import division, print_function

import os
import time

import cv2
import numpy as np

from application_util import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from detectors import create_detector
from pipeline.config import load_config
from reid import create_reid

import deep_sort_app


def run_sequence(sequence_dir, config, output_file=None, display=False,
                 save_detections=None, config_path=None):
    """
    Run tracker on a MOT sequence with live detection + ReID.

    Returns
    -------
    dict with keys: results, fps, num_frames, sequence_name
    """
    seq_info = deep_sort_app.gather_sequence_info(sequence_dir, detection_file=None)
    video_name = seq_info["sequence_name"]

    base_cfg = load_config(config_path, video_name=video_name)
    cfg = dict(base_cfg)
    if config:
        cfg.update(config)

    detector = create_detector(
        cfg["detector"],
        conf_threshold=cfg.get("detector_conf", 0.25),
        imgsz=cfg.get("imgsz", 640),
        device=cfg.get("device"))
    encoder = create_reid(
        cfg["reid"],
        device=cfg.get("device"))

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", cfg["max_cosine_distance"], cfg.get("nn_budget"))
    tracker = Tracker(
        metric,
        max_iou_distance=cfg.get("max_iou_distance", 0.7),
        max_age=cfg.get("max_age", 30),
        n_init=cfg.get("n_init", 3))

    results = []
    detection_rows = []
    frame_times = []

    min_frame = seq_info["min_frame_idx"]
    max_frame = seq_info["max_frame_idx"]

    for frame_idx in range(min_frame, max_frame + 1):
        if frame_idx not in seq_info["image_filenames"]:
            continue
        t0 = time.time()
        image = cv2.imread(seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)

        raw_dets = detector.detect(image)
        if not raw_dets:
            detections = []
        else:
            boxes = np.array([d[0] for d in raw_dets])
            confs = np.array([d[1] for d in raw_dets])
            features = encoder.encode(image, boxes)
            detections = []
            for (tlwh, conf), feat in zip(raw_dets, features):
                if tlwh[3] < cfg.get("min_detection_height", 0):
                    continue
                if conf < cfg.get("min_confidence", 0.3):
                    continue
                detections.append(Detection(tlwh, conf, feat))
                if save_detections is not None:
                    row = np.zeros(10 + features.shape[1])
                    row[0] = frame_idx
                    row[2:6] = tlwh
                    row[6] = conf
                    row[10:] = feat
                    detection_rows.append(row)

        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        if len(boxes) > 0:
            indices = preprocessing.non_max_suppression(
                boxes, cfg.get("nms_max_overlap", 0.7), scores)
            detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        frame_times.append(time.time() - t0)
        if frame_idx % 50 == 0:
            print("Frame %05d/%05d" % (frame_idx, max_frame))

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            for row in results:
                f.write("%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1\n" % tuple(row))

    if save_detections and detection_rows:
        os.makedirs(os.path.dirname(save_detections) or ".", exist_ok=True)
        np.save(save_detections, np.asarray(detection_rows), allow_pickle=False)

    avg_fps = len(frame_times) / sum(frame_times) if frame_times else 0.0
    return {
        "results": results,
        "fps": avg_fps,
        "num_frames": len(frame_times),
        "sequence_name": video_name,
    }


def run_mot_directory(mot_dir, config, output_dir, config_path=None):
    """Run tracker on all sequences in a MOT directory."""
    os.makedirs(output_dir, exist_ok=True)
    all_stats = []
    for sequence in sorted(os.listdir(mot_dir)):
        seq_path = os.path.join(mot_dir, sequence)
        if not os.path.isdir(seq_path) or not os.path.isdir(os.path.join(seq_path, "img1")):
            continue
        print("Running sequence: %s" % sequence)
        cfg = load_config(config_path, video_name=sequence)
        cfg.update(config or {})
        out_file = os.path.join(output_dir, "%s.txt" % sequence)
        stats = run_sequence(
            seq_path, cfg, output_file=out_file, config_path=config_path)
        all_stats.append(stats)
        print("  FPS: %.2f" % stats["fps"])
    return all_stats
