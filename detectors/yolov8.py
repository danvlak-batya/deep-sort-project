from ultralytics import YOLO

from detectors.base import BaseDetector


class YOLOv8Detector(BaseDetector):
  """YOLOv8 person detector (Ultralytics)."""

  def __init__(self, model_name="yolov8n.pt", conf_threshold=0.25, imgsz=640, device=None):
    self.model = YOLO(model_name)
    self.conf_threshold = conf_threshold
    self.imgsz = imgsz
    self.device = device

  @property
  def name(self):
    return "yolov8n"

  def detect(self, image):
    results = self.model.predict(
        source=image,
        conf=self.conf_threshold,
        imgsz=self.imgsz,
        classes=[self.PERSON_CLASS_ID],
        verbose=False,
        device=self.device)
    detections = []
    if not results:
      return detections
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
      return detections
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    for (x1, y1, x2, y2), conf in zip(xyxy, confs):
      tlwh = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
      detections.append((tlwh, float(conf)))
    return detections
