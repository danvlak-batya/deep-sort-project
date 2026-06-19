"""Configuration loading for tracker pipeline."""
import os

import yaml

DEFAULT_CONFIG = {
    "detector": "yolov8n",
    "reid": "osnet_x0_25",
    "detector_conf": 0.25,
    "imgsz": 640,
    "min_confidence": 0.3,
    "nms_max_overlap": 0.7,
    "min_detection_height": 0,
    "max_cosine_distance": 0.2,
    "nn_budget": 100,
    "max_iou_distance": 0.7,
    "max_age": 30,
    "n_init": 3,
    "device": None,
}


def deep_merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path=None, video_name=None):
    """Load default config with optional file and per-video overrides."""
    config = dict(DEFAULT_CONFIG)
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = deep_merge(config, yaml.safe_load(f) or {})

    if video_name:
        video_cfg_path = os.path.join(
            os.path.dirname(config_path or "configs/default.yaml"),
            "videos", "%s.yaml" % video_name)
        if os.path.exists(video_cfg_path):
            with open(video_cfg_path, "r", encoding="utf-8") as f:
                config = deep_merge(config, yaml.safe_load(f) or {})
    return config
