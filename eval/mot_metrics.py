"""MOT evaluation utilities using TrackEval (HOTA)."""
from __future__ import print_function

import os
import shutil
import tempfile


EVAL_SEQUENCES = [
    "TUD-Campus",
    "TUD-Stadtmitte",
    "KITTI-17",
    "PETS09-S2L1",
    "MOT16-09",
    "MOT16-11",
]


def _find_sequence_path(mot_root, sequence_name):
    search_roots = [mot_root]
    for sub in ("MOT15", "MOT16"):
        candidate_root = os.path.join(mot_root, sub)
        if os.path.isdir(candidate_root):
            search_roots.append(candidate_root)

    for root in search_roots:
        for split in ("train", "test", ""):
            if split:
                candidate = os.path.join(root, split, sequence_name)
            else:
                candidate = os.path.join(root, sequence_name)
            if os.path.isdir(candidate):
                return candidate
    raise FileNotFoundError("Sequence %s not found under %s" % (sequence_name, mot_root))


def _prepare_trackeval_folders(mot_root, results_dir, sequences, tracker_name="deep_sort"):
    """Build folder layout expected by TrackEval."""
    tmp = tempfile.mkdtemp(prefix="trackeval_")
    gt_root = os.path.join(tmp, "gt")
    trackers_root = os.path.join(tmp, "trackers")
    tracker_dir = os.path.join(trackers_root, tracker_name, "data")
    os.makedirs(tracker_dir, exist_ok=True)

    for seq in sequences:
        seq_path = _find_sequence_path(mot_root, seq)
        gt_seq_dir = os.path.join(gt_root, seq, "gt")
        os.makedirs(gt_seq_dir, exist_ok=True)
        gt_src = os.path.join(seq_path, "gt", "gt.txt")
        if os.path.exists(gt_src):
            shutil.copy(gt_src, os.path.join(gt_seq_dir, "gt.txt"))

        result_file = os.path.join(results_dir, "%s.txt" % seq)
        if os.path.exists(result_file):
            shutil.copy(result_file, os.path.join(tracker_dir, "%s.txt" % seq))

    return tmp, gt_root, trackers_root


def evaluate_hota(mot_root, results_dir, sequences=None, tracker_name="deep_sort"):
    """
    Compute HOTA and related MOT metrics via TrackEval.

    Returns
    -------
    dict
        Per-sequence and mean HOTA scores.
    """
    sequences = sequences or EVAL_SEQUENCES
    tmp, gt_root, trackers_root = _prepare_trackeval_folders(
        mot_root, results_dir, sequences, tracker_name)

    try:
        import trackeval

        eval_config = trackeval.Evaluator.get_default_eval_config()
        eval_config["PRINT_RESULTS"] = False
        eval_config["PRINT_ONLY_COMBINED"] = False
        eval_config["DISPLAY_LESS_PROGRESS"] = True

        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dataset_config["GT_FOLDER"] = gt_root
        dataset_config["TRACKERS_FOLDER"] = trackers_root
        dataset_config["TRACKERS_TO_EVAL"] = [tracker_name]
        dataset_config["BENCHMARK"] = ""
        dataset_config["SPLIT_TO_EVAL"] = ""
        dataset_config["SEQ_INFO"] = {
            seq: {"seq_length": -1} for seq in sequences if os.path.exists(
                os.path.join(trackers_root, tracker_name, "data", "%s.txt" % seq))}
        dataset_config["CLASSES_TO_EVAL"] = ["pedestrian"]

        metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"]}
        evaluator = trackeval.Evaluator(eval_config)
        dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
        metrics = [trackeval.metrics.HOTA(metrics_config)]
        results = evaluator.evaluate([dataset], metrics)

        output = {"per_sequence": {}, "mean_hota": 0.0}
        hota_vals = []
        tracker_res = results[dataset.name][tracker_name]
        for seq in sequences:
            if seq not in tracker_res:
                continue
            hota = float(tracker_res[seq]["HOTA"]["HOTA"])
            output["per_sequence"][seq] = {"HOTA": hota}
            hota_vals.append(hota)
        output["mean_hota"] = float(sum(hota_vals) / len(hota_vals)) if hota_vals else 0.0
        return output
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def save_metrics_json(metrics, path):
    import json
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
