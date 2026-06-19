"""Shared image utilities for detection and ReID."""
import cv2
import numpy as np


def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box (x, y, w, h)."""
    bbox = np.array(bbox, dtype=np.float64)
    if patch_shape is not None:
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int64)
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    patch = image[sy:ey, sx:ex]
    if patch.size == 0:
        return None
    patch = cv2.resize(patch, tuple(patch_shape[::-1]))
    return patch


def iou_matrix(boxes_a, boxes_b):
    """Compute IoU between two sets of boxes in (x, y, w, h) format."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    boxes_a = np.asarray(boxes_a, dtype=np.float64)
    boxes_b = np.asarray(boxes_b, dtype=np.float64)

    a_x2 = boxes_a[:, 0] + boxes_a[:, 2]
    a_y2 = boxes_a[:, 1] + boxes_a[:, 3]
    b_x2 = boxes_b[:, 0] + boxes_b[:, 2]
    b_y2 = boxes_b[:, 1] + boxes_b[:, 3]

    ious = np.zeros((len(boxes_a), len(boxes_b)))
    for i, (x1, y1, x2, y2) in enumerate(zip(boxes_a[:, 0], boxes_a[:, 1], a_x2, a_y2)):
        xx1 = np.maximum(x1, boxes_b[:, 0])
        yy1 = np.maximum(y1, boxes_b[:, 1])
        xx2 = np.minimum(x2, b_x2)
        yy2 = np.minimum(y2, b_y2)
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_a = (x2 - x1) * (y2 - y1)
        area_b = (boxes_b[:, 2]) * (boxes_b[:, 3])
        union = area_a + area_b - inter
        ious[i] = np.where(union > 0, inter / union, 0)
    return ious
