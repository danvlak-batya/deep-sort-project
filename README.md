# Deep SORT — Extended Final Project

Extended implementation of [Deep SORT](https://github.com/nwojke/deep_sort) with modern person detectors and ReID models for MOT Challenge evaluation.

## Features

- **3 detectors**: YOLOv8n (Ultralytics), YOLOv5s (torch.hub), RT-DETR-R18 (HuggingFace)
- **3 ReID models**: OSNet x0.25, ResNet50-IBN (torchreid), fast-reid SBS
- Unified tracking pipeline with per-video YAML configs
- HOTA evaluation via [TrackEval](https://github.com/JonathonLuiten/TrackEval)
- Detector F1 and ReID clustering metrics
- Overlay video generation
- Google Colab notebook for reproducible runs

## Evaluation sequences

- TUD-Campus, TUD-Stadtmitte, KITTI-17, PETS09-S2L1 (MOT15)
- MOT16-09, MOT16-11 (MOT16)

## Quick start (Google Colab)

See [`notebooks/DeepSORT_Colab.ipynb`](notebooks/DeepSORT_Colab.ipynb) for full instructions.

```python
!git clone https://github.com/<your-user>/deep-sort-project.git
%cd deep-sort-project
!pip install -q -r requirements.txt
```

Mount MOT data on Google Drive under `MyDrive/MOT/` with standard MOTChallenge layout.

## Run modern tracker

```bash
# Single sequence
python run_tracker.py \
  --sequence_dir /path/to/MOT16/train/MOT16-09 \
  --config configs/default.yaml \
  --detector yolov8n \
  --reid osnet_x0_25 \
  --output_file results/modern/MOT16-09.txt

# All sequences in a directory
python run_tracker.py \
  --mot_dir /path/to/MOT16/train \
  --output_dir results/modern
```

## Baseline (original DeepSORT)

The original tracker entry point is preserved:

```bash
python deep_sort_app.py \
  --sequence_dir ./MOT16/train/MOT16-09 \
  --detection_file ./resources/detections/MOT16-09.npy \
  --min_confidence=0.3 \
  --display=False
```

Pre-generated detections and the TensorFlow ReID model are available from the [original release](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp).

## Evaluation

```bash
# Full pipeline + HOTA
python eval/run_mot.py --mot_dir /path/to/MOT --output_dir results/modern

# Detector F1 vs ground truth
python eval/eval_detector.py --mot_dir /path/to/MOT --detector yolov8n

# ReID quality with GT boxes
python eval/eval_reid.py --mot_dir /path/to/MOT --reid osnet_x0_25

# Parameter grid search (subset of videos)
python eval/tune_params.py --mot_dir /path/to/MOT

# Generate overlay videos
python tools/generate_overlays.py \
  --mot_dir /path/to/MOT \
  --results_dir results/modern \
  --output_dir results/overlays
```

## Configuration

Default settings: [`configs/default.yaml`](configs/default.yaml)

Per-video overrides: [`configs/videos/`](configs/videos/)

Recommended real-time combo (≥5 FPS on Colab T4): **YOLOv8n + osnet_x0_25**.

## Project structure

```
detectors/          # YOLOv8, YOLOv5, RT-DETR backends
reid/               # torchreid and fast-reid backends
pipeline/           # Unified tracker + config loader
eval/               # HOTA, detector F1, ReID metrics, tuning
configs/            # Default and per-video parameters
tools/              # Overlay generation (+ original scripts)
notebooks/          # Colab notebook
report/             # Project report
deep_sort/          # Original DeepSORT core (unchanged)
```

## Report

See [`report/report.md`](report/report.md) for methodology, parameter evolution, and results tables.

## Citing DeepSORT

```bibtex
@inproceedings{Wojke2017simple,
  title={Simple Online and Realtime Tracking with a Deep Association Metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={ICIP}, year={2017}
}
```
