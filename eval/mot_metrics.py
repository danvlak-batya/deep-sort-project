"""MOT evaluation utilities using TrackEval (HOTA)."""
from __future__ import print_function

import os
import shutil
import tempfile

from utils.mot_paths import find_sequence_dir, get_gt_file, list_image_filenames


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


def _prepare_trackeval_folders(mot_root, results_dir, sequences, tracker_name="deep_sort"):
    """Build folder layout for TrackEval with SKIP_SPLIT_FOL=True.

    Layout:
      gt/<SEQ>/gt/gt.txt
      trackers/<TRACKER>/data/<SEQ>.txt
    """
    tmp = tempfile.mkdtemp(prefix="trackeval_")
    gt_root = os.path.join(tmp, "gt")
    trackers_root = os.path.join(tmp, "trackers")
    tracker_dir = os.path.join(trackers_root, tracker_name, "data")
    os.makedirs(tracker_dir, exist_ok=True)

    prepared = []
    seq_info = {}
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

        gt_seq_dir = os.path.join(gt_root, seq, "gt")
        os.makedirs(gt_seq_dir, exist_ok=True)
        gt_dst = os.path.join(gt_seq_dir, "gt.txt")
        shutil.copy(gt_src, gt_dst)
        shutil.copy(result_file, os.path.join(tracker_dir, "%s.txt" % seq))

        num_frames = len(list_image_filenames(seq_path))
        seq_info[seq] = max(num_frames, 1)
        prepared.append(seq)
        print("Prepared %s: %d frames -> %s" % (seq, num_frames, gt_dst))

    if not prepared:
        raise FileNotFoundError(
            "No sequences ready for HOTA. Check results_dir (.txt files) and GT folders.")

    return tmp, gt_root, trackers_root, prepared, seq_info


def _extract_hota_score(seq_result, class_name="pedestrian"):
    """Read scalar HOTA from TrackEval per-sequence result dict."""
    import numpy as np

    if class_name not in seq_result:
        class_name = next(iter(seq_result))
    hota_val = seq_result[class_name]["HOTA"]["HOTA"]
    if hasattr(hota_val, "__len__"):
        return float(np.mean(hota_val))
    return float(hota_val)


def evaluate_hota(mot_root, results_dir, sequences=None, tracker_name="deep_sort"):
    """
    Compute HOTA via TrackEval.

    Returns dict with per_sequence HOTA and mean_hota.
    """
    sequences = sequences or EVAL_SEQUENCES
    tmp, gt_root, trackers_root, prepared, seq_info = _prepare_trackeval_folders(
        mot_root, results_dir, sequences, tracker_name)

    try:
        import trackeval

        eval_config = trackeval.Evaluator.get_default_eval_config()
        eval_config["PRINT_RESULTS"] = False
        eval_config["PRINT_ONLY_COMBINED"] = False
        eval_config["DISPLAY_LESS_PROGRESS"] = True
        eval_config["BREAK_ON_ERROR"] = True

        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dataset_config["GT_FOLDER"] = gt_root
        dataset_config["TRACKERS_FOLDER"] = trackers_root
        dataset_config["TRACKERS_TO_EVAL"] = [tracker_name]
        dataset_config["BENCHMARK"] = ""
        dataset_config["SPLIT_TO_EVAL"] = ""
        dataset_config["SEQ_INFO"] = seq_info
        dataset_config["CLASSES_TO_EVAL"] = ["pedestrian"]
        dataset_config["SKIP_SPLIT_FOL"] = True
        dataset_config["DO_PREPROC"] = True

        print("TrackEval GT_FOLDER:", gt_root)
        print("SEQ_INFO:", seq_info)

        evaluator = trackeval.Evaluator(eval_config)
        dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
        metrics = [trackeval.metrics.HOTA({"METRICS": ["HOTA"], "THRESHOLD": 0.5})]
        eval_output = evaluator.evaluate([dataset], metrics)
        if isinstance(eval_output, tuple):
            results = eval_output[0]
        else:
            results = eval_output

        dataset_key = "MotChallenge2DBox"
        if dataset_key not in results:
            dataset_key = next(iter(results))

        output = {"per_sequence": {}, "mean_hota": 0.0}
        hota_vals = []
        tracker_res = results[dataset_key][tracker_name]
        for seq in prepared:
            if seq not in tracker_res:
                print("Warning: no HOTA result for", seq)
                continue
            hota = _extract_hota_score(tracker_res[seq])
            output["per_sequence"][seq] = {"HOTA": hota}
            hota_vals.append(hota)
        output["mean_hota"] = float(sum(hota_vals) / len(hota_vals)) if hota_vals else 0.0
        if not hota_vals:
            raise RuntimeError("TrackEval ran but returned no HOTA values.")
        return output
    except Exception as exc:
        raise RuntimeError("HOTA evaluation failed: %s" % exc) from exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def save_metrics_json(metrics, path):
    import json
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
