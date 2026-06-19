from detectors.base import BaseDetector
from detectors.yolov8 import YOLOv8Detector
from detectors.yolov5 import YOLOv5Detector
from detectors.rtdetr import RTDETRDetector

DETECTOR_REGISTRY = {
    "yolov8n": YOLOv8Detector,
    "yolov5s": YOLOv5Detector,
    "rtdetr_r18": RTDETRDetector,
}


def create_detector(name, **kwargs):
    """Factory for detector backends."""
    if name not in DETECTOR_REGISTRY:
        raise ValueError(
            "Unknown detector '%s'. Choose from: %s" % (
                name, ", ".join(DETECTOR_REGISTRY)))
    return DETECTOR_REGISTRY[name](**kwargs)


def list_detectors():
    return list(DETECTOR_REGISTRY.keys())
