"""MOT evaluation utilities using TrackEval (HOTA)."""
from __future__ import print_function

import os
import shutil
import tempfile

from utils.mot_paths import find_sequence_dir, get_gt_file


EVAL_SEQUENCES = [
    "TUD-Campus",
    "TUD-Stadtmitte",
    "KITTI-17",
    "PETS09-S2L1",
    "MOT16-09",
    "MOT16-11",
]


def _find_sequence_path(mot_root, sequence_name):
    return find_sequence_dir(mot_root, sequence_name)


MOT15_SEQUENCES = {"TUD-Campus", "TUD-Stadtmitte", "KITTI-17", "PETS09-S2L1"}
MOT16_SEQUENCES = {"MOT16-09", "MOT16-11"}

# TrackEval expects: GT_FOLDER / BENCHMARK / SPLIT / SEQ / gt / gt.txt
TRACKEVAL_BENCHMARK = "MOT"
TRACKEVAL_SPLIT = "train"


def _prepare_trackeval_folders(mot_root, results_dir, sequences, tracker_name="deep_sort"):
    """Build folder layout expected by TrackEval MotChallenge2DBox."""
    tmp = tempfile.mkdtemp(prefix="trackeval_")
    gt_root = os.path.join(tmp, "gt")
    trackers_root = os.path.join(tmp, "trackers")
    tracker_dir = os.path.join(trackers_root, tracker_name, "data")
    os.makedirs(tracker_dir, exist_ok=True)

    prepared = []
    for seq in sequences:
        result_file = os.path.join(results_dir, "%s.txt" % seq)
        if not os.path.exists(result_file):
            print("Skip %s: no tracker results at %s" % (seq, result_file))
            continue
        try:
            seq_path = _find_sequence_path(mot_root, seq)
        except FileNotFoundError:
            print("Skip %s: sequence folder not found" % seq)
            continue

        gt_src = get_gt_file(seq_path)
        if not os.path.exists(gt_src):
            print("Skip %s: GT not found at %s" % (seq, gt_src))
            continue

        gt_seq_dir = os.path.join(
            gt_root, TRACKEVAL_BENCHMARK, TRACKEVAL_SPLIT, seq, "gt")
        os.makedirs(gt_seq_dir, exist_ok=True)
        shutil.copy(gt_src, os.path.join(gt_seq_dir, "gt.txt"))
        shutil.copy(result_file, os.path.join(tracker_dir, "%s.txt" % seq))
        prepared.append(seq)

    if not prepared:
        raise FileNotFoundError(
            "No sequences ready for HOTA. Check results_dir (.txt files) and GT folders.")

    return tmp, gt_root, trackers_root, prepared


def evaluate_hota(mot_root, results_dir, sequences=None, tracker_name="deep_sort"):
    """
    Compute HOTA and related MOT metrics via TrackEval.

    Returns
    -------
    dict
        Per-sequence and mean HOTA scores.
    """
    sequences = sequences or EVAL_SEQUENCES
    tmp, gt_root, trackers_root, prepared = _prepare_trackeval_folders(
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
        dataset_config["BENCHMARK"] = TRACKEVAL_BENCHMARK
        dataset_config["SPLIT_TO_EVAL"] = TRACKEVAL_SPLIT
        dataset_config["SEQ_INFO"] = {seq: {"seq_length": -1} for seq in prepared}
        dataset_config["CLASSES_TO_EVAL"] = ["pedestrian"]
        dataset_config["SKIP_SPLIT_FOL"] = True

        metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"]}
        evaluator = trackeval.Evaluator(eval_config)
        dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
        metrics = [trackeval.metrics.HOTA(metrics_config)]
        results = evaluator.evaluate([dataset], metrics)

        output = {"per_sequence": {}, "mean_hota": 0.0}
        hota_vals = []
        tracker_res = results[dataset.name][tracker_name]
        for seq in prepared:
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
