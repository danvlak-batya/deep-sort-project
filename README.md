# Deep SORT — Extended Final Project

Extended [Deep SORT](https://github.com/nwojke/deep_sort) for HSE Deep Learning course.

## Deliverables

- 3 detectors + 3 ReID models (selectable)
- HOTA evaluation on 6 MOT sequences
- Colab notebook: [`notebooks/DeepSORT_Colab.ipynb`](notebooks/DeepSORT_Colab.ipynb)
- Report: [`report/report.md`](report/report.md)
- Overlays: baseline + best

## Data layout (Google Drive)

```
MyDrive/Videos-CV/
  Kitti-17/img/   gt/   det/
  MOT16-09/img/   gt/   det/
  ...
```

## Colab quick start

1. Open `notebooks/DeepSORT_Colab.ipynb` in Colab (GPU runtime)
2. Cell 2 runs `python tools/install_colab_deps.py` (do **not** use plain `pip install -r requirements.txt` — torchreid fails on Colab Python 3.12)
3. Run cells 1–6 in order

Recommended combo: **YOLOv8n + osnet_x0_25**

## Models

| Detectors | ReID |
|-----------|------|
| yolov8n (Ultralytics) | osnet_x0_25 (timm) |
| yolov5s (torch.hub) | resnet50_ibn (timm) |
| rtdetr_r18 (HuggingFace) | fastreid_sbs / osnet_x1_0 (timm) |

## Evaluation commands (Colab)

```bash
python eval/run_benchmark.py --mot_root $DATA_ROOT --output_dir results/modern
python eval/run_mot.py --mot_dir $DATA_ROOT --output_dir results/modern --skip_tracking
python eval/eval_detector.py --mot_dir $DATA_ROOT --detector yolov8n
python eval/eval_reid.py --mot_dir $DATA_ROOT --reid osnet_x0_25
python tools/generate_overlays.py --mot_dir $DATA_ROOT --results_dir results/modern --output_dir results/overlays
```

## Baseline

Download pre-generated detections from [original DeepSORT release](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp), then:

```bash
python eval/run_baseline.py --mot_root $DATA_ROOT --detections_dir resources/detections/
```

## Structure

```
detectors/  reid/  pipeline/  eval/  configs/  tools/  notebooks/  report/
deep_sort/  deep_sort_app.py  run_tracker.py   # original + entry points
```
