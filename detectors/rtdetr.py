import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from detectors.base import BaseDetector

# COCO person class id
COCO_PERSON_ID = 1


class RTDETRDetector(BaseDetector):
  """RT-DETR-R18 person detector (HuggingFace Transformers)."""

  def __init__(self, conf_threshold=0.3, device=None):
    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "PekingU/rtdetr_r18vd"
    self.processor = RTDetrImageProcessor.from_pretrained(model_id)
    self.model = RTDetrForObjectDetection.from_pretrained(model_id)
    self.model.to(self.device)
    self.model.eval()
    self.conf_threshold = conf_threshold

  @property
  def name(self):
    return "rtdetr_r18"

  def detect(self, image):
    import numpy as np
    from PIL import Image

    rgb = Image.fromarray(image[:, :, ::-1])
    inputs = self.processor(images=rgb, return_tensors="pt")
    inputs = {k: v.to(self.device) for k, v in inputs.items()}
    with torch.no_grad():
      outputs = self.model(**inputs)
    target_sizes = torch.tensor([[rgb.size[1], rgb.size[0]]], device=self.device)
    results = self.processor.post_process_object_detection(
        outputs, threshold=self.conf_threshold, target_sizes=target_sizes)[0]

    detections = []
    for score, label, box in zip(
        results["scores"].cpu().numpy(),
        results["labels"].cpu().numpy(),
        results["boxes"].cpu().numpy()):
      if int(label) != COCO_PERSON_ID:
        continue
      x1, y1, x2, y2 = box
      tlwh = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
      detections.append((tlwh, float(score)))
    return detections
