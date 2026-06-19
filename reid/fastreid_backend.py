import os

import cv2
import numpy as np
import torch

from reid.base import BaseReID
from utils.image import extract_image_patch

FASTREID_WEIGHTS_URL = (
    "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/"
    "mot17_sbs_S50.pth")


class FastReIDEncoder(BaseReID):
    """Person ReID via fast-reid (JDAI-CV). Falls back to torchreid if unavailable."""

    def __init__(self, weights_path=None, batch_size=32, device=None):
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = (256, 128)
        self._feature_dim = 2048
        self._use_fastreid = False
        self._fallback = None

        try:
            from fastreid.config import get_cfg
            from fastreid.engine import DefaultPredictor

            cfg = get_cfg()
            cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
            cfg.MODEL.BACKBONE.DEPTH = "50x"
            cfg.MODEL.BACKBONE.WITH_IBN = True
            cfg.MODEL.HEADS.NAME = "EmbeddingHead"
            cfg.MODEL.HEADS.EMBEDDING_DIM = 0
            cfg.MODEL.HEADS.NECK_FEAT = "before"
            cfg.MODEL.HEADS.POOL_LAYER = "gempool"
            cfg.MODEL.WEIGHTS = weights_path or self._ensure_weights()
            cfg.MODEL.DEVICE = self.device
            cfg.INPUT.SIZE_TEST = list(self.input_size)
            self.predictor = DefaultPredictor(cfg)
            self._use_fastreid = True
        except Exception as exc:
            print("fast-reid unavailable (%s), using torchreid resnet50_ibn_a fallback." % exc)
            from reid.torchreid_backend import TorchReIDEncoder
            self._fallback = TorchReIDEncoder(
                model_name="resnet50_ibn_a", batch_size=batch_size, device=self.device)
            self._feature_dim = self._fallback.feature_dim

    @staticmethod
    def _ensure_weights():
        import urllib.request
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "fastreid")
        os.makedirs(cache_dir, exist_ok=True)
        weights_path = os.path.join(cache_dir, "mot17_sbs_S50.pth")
        if not os.path.exists(weights_path):
            print("Downloading fast-reid weights...")
            urllib.request.urlretrieve(FASTREID_WEIGHTS_URL, weights_path)
        return weights_path

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def name(self):
        return "fastreid_sbs"

    def encode(self, image, boxes):
        if self._fallback is not None:
            return self._fallback.encode(image, boxes)

        boxes = np.asarray(boxes, dtype=np.float64)
        if len(boxes) == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        features = np.zeros((len(boxes), self.feature_dim), dtype=np.float32)
        patches = []
        valid_idx = []
        for i, box in enumerate(boxes):
            patch = extract_image_patch(image, box, self.input_size)
            if patch is None:
                continue
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patches.append(patch)
            valid_idx.append(i)

        if not patches:
            return features

        for start in range(0, len(patches), self.batch_size):
            batch_patches = patches[start:start + self.batch_size]
            batch_idx = valid_idx[start:start + len(batch_patches)]
            for j, patch in enumerate(batch_patches):
                outputs = self.predictor(patch)
                if isinstance(outputs, torch.Tensor):
                    emb = outputs
                else:
                    emb = outputs["features"] if isinstance(outputs, dict) else outputs
                emb = torch.nn.functional.normalize(emb.flatten(), p=2, dim=0)
                features[batch_idx[j]] = emb.cpu().numpy()
        return features
