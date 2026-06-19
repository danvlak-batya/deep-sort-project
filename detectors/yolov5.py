import torch

from detectors.base import BaseDetector


class YOLOv5Detector(BaseDetector):
  """YOLOv5 person detector (Ultralytics YOLOv5 via torch.hub)."""

  def __init__(self, model_name="yolov5s", conf_threshold=0.25, imgsz=640, device=None):
    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    self.model = torch.hub.load(
        "ultralytics/yolov5", model_name, pretrained=True, trust_repo=True)
    self.model.to(self.device)
    self.model.conf = conf_threshold
    self.model.classes = [self.PERSON_CLASS_ID]
    self.imgsz = imgsz
    self._model_name = model_name

  @property
  def name(self):
    return "yolov5s"

  def detect(self, image):
    results = self.model(image, size=self.imgsz)
    detections = []
    pred = results.pred[0]
    if pred is None or len(pred) == 0:
      return detections
    for *xyxy, conf, cls in pred.cpu().numpy():
      if int(cls) != self.PERSON_CLASS_ID:
        continue
      x1, y1, x2, y2 = xyxy
      tlwh = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
      detections.append((tlwh, float(conf)))
    return detections
