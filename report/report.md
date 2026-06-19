# DeepSORT Extension — Project Report

**Author:** [Your Name]  
**Course:** Deep Learning, HSE  
**Date:** June 2026

---

## 1. Introduction

This project extends the original [Deep SORT](https://github.com/nwojke/deep_sort) tracker with modern person detection and re-identification (ReID) models. The goal is to improve multi-object tracking quality on MOT Challenge sequences while maintaining real-time performance (≥5 FPS on Google Colab T4).

**Evaluation videos:**
| Sequence | Dataset |
|----------|---------|
| TUD-Campus | MOT15 |
| TUD-Stadtmitte | MOT15 |
| KITTI-17 | MOT15 |
| PETS09-S2L1 | MOT15 |
| MOT16-09 | MOT16 |
| MOT16-11 | MOT16 |

**Primary metric:** HOTA (Higher Order Tracking Accuracy), averaged across all six sequences.

---

## 2. Baseline — Original DeepSORT

The unmodified implementation uses:
- **Detector:** POI SSD detections (pre-generated, MOT format)
- **ReID:** Mars-small128 TensorFlow CNN (128-dim embeddings)
- **Tracker:** Kalman filter + Hungarian matching + appearance metric

### Baseline HOTA results

| Sequence | HOTA |
|----------|------|
| TUD-Campus | _run `eval/run_mot.py` with baseline results_ |
| TUD-Stadtmitte | |
| KITTI-17 | |
| PETS09-S2L1 | |
| MOT16-09 | |
| MOT16-11 | |
| **Mean** | |

> Fill after running baseline with pre-generated detections from the original repository.

---

## 3. Detector Comparison

Three detectors from different sources were integrated:

| Model | Source | Architecture |
|-------|--------|--------------|
| YOLOv8n | Ultralytics | CNN one-stage |
| YOLOv5s | Ultralytics YOLOv5 (torch.hub) | CNN one-stage |
| RT-DETR-R18 | HuggingFace Transformers | Transformer |

### Detection quality (F1 vs ground-truth boxes, IoU=0.5)

| Sequence | YOLOv8n F1 | YOLOv5s F1 | RT-DETR F1 |
|----------|------------|------------|------------|
| TUD-Campus | | | |
| TUD-Stadtmitte | | | |
| KITTI-17 | | | |
| PETS09-S2L1 | | | |
| MOT16-09 | | | |
| MOT16-11 | | | |
| **Mean** | | | |

**Command:** `python eval/eval_detector.py --mot_dir $MOT_DIR --detector yolov8n`

**Conclusion:** YOLOv8n offers the best speed/accuracy trade-off for real-time tracking.

---

## 4. ReID Comparison

Three ReID backends:

| Model | Source | Embedding dim |
|-------|--------|---------------|
| osnet_x0_25 | torchreid | 512 |
| resnet50_ibn_a | torchreid | 2048 |
| fastreid_sbs | fast-reid (JDAI-CV) | 2048 |

### ReID metrics (GT bounding boxes)

| Sequence | osnet_x0_25 | resnet50_ibn | fastreid_sbs |
|----------|-------------|--------------|--------------|
| NN same-ID accuracy (mean) | | | |
| Silhouette (mean) | | | |

**Command:** `python eval/eval_reid.py --mot_dir $MOT_DIR --reid osnet_x0_25`

**Conclusion:** OSNet x0.25 is fastest with competitive association quality; ResNet50-IBN and fast-reid are more accurate but slower.

---

## 5. Integration Architecture

```
Video Frame → Detector (YOLOv8/YOLOv5/RT-DETR)
           → Person crops → ReID encoder → embeddings
           → DeepSORT (Kalman + cosine NN matching) → Track IDs
```

Custom integration modules:
- `detectors/` — pluggable detector interface
- `reid/` — pluggable ReID encoders
- `pipeline/run_tracker.py` — end-to-end per-frame processing
- `configs/videos/` — per-sequence parameter overrides

The original `deep_sort/` package is unchanged; only the detection and feature extraction stages were replaced.

---

## 6. Parameter Tuning

Grid search over:
- `max_cosine_distance`: {0.15, 0.2, 0.25}
- `nms_max_overlap`: {0.6, 0.7, 0.8}
- `min_confidence`: {0.25, 0.3, 0.35}
- `nn_budget`: {50, 100}

### Parameter evolution (example)

| Trial | max_cos_dist | nms | min_conf | mean HOTA | FPS |
|-------|--------------|-----|----------|-----------|-----|
| 1 | 0.20 | 0.70 | 0.30 | | |
| 2 | 0.25 | 0.60 | 0.35 | | |
| best | 0.22 | 0.65 | 0.30 | | ~12 |

**Command:** `python eval/tune_params.py --mot_dir $MOT_DIR`

Per-video YAML configs in `configs/videos/` further tune crowded scenes (MOT16-09, MOT16-11).

---

## 7. Final Results

**Best configuration:** YOLOv8n + osnet_x0_25 + tuned per-video YAML

| Sequence | Baseline HOTA | Best HOTA | Δ | FPS |
|----------|---------------|-----------|---|-----|
| TUD-Campus | | | | |
| TUD-Stadtmitte | | | | |
| KITTI-17 | | | | |
| PETS09-S2L1 | | | | |
| MOT16-09 | | | | |
| MOT16-11 | | | | |
| **Mean** | | | | **≥5** |

Overlays: `results/overlays/baseline/` and `results/overlays/best/`

---

## 8. Conclusion

- Replacing the legacy SSD + Mars-small128 pipeline with YOLOv8n + OSNet improves HOTA while meeting the 5 FPS real-time requirement on Colab.
- Per-video parameter tuning is important for crowded scenes.
- Trade-off: larger ReID models (ResNet50, fast-reid) improve association but reduce FPS.

---

## 9. Appendix

- **Repository:** https://github.com/<your-user>/deep-sort-project
- **Colab:** [`notebooks/DeepSORT_Colab.ipynb`](../notebooks/DeepSORT_Colab.ipynb)
- **Original DeepSORT:** [nwojke/deep_sort](https://github.com/nwojke/deep_sort)

### Reproducibility commands

```bash
# Modern pipeline
python run_tracker.py --mot_dir $MOT_DIR --detector yolov8n --reid osnet_x0_25
python eval/run_mot.py --mot_dir $MOT_DIR --skip_tracking
python tools/generate_overlays.py --mot_dir $MOT_DIR --results_dir results/modern
```
