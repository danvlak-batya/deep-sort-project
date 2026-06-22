import numpy as np
import torch
import torchreid
from PIL import Image

from reid.base import BaseReID
from utils.image import extract_image_patch


class TorchReIDEncoder(BaseReID):
    """Person ReID via torchreid (OSNet, ResNet-IBN, etc.)."""

    def __init__(self, model_name="osnet_x0_25", batch_size=32, device=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1,
            pretrained=True)
        self.model.eval()
        self.model.to(self.device)
        self._feature_dim = self.model.feature_dim
        self.input_size = (256, 128)
        self.transform = torchreid.data.transforms.build_transforms(
            height=self.input_size[0],
            width=self.input_size[1],
            is_train=False)[1]

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def name(self):
        return self.model_name

    def _preprocess_patches(self, image, boxes):
        patches = []
        valid_idx = []
        for i, box in enumerate(boxes):
            patch = extract_image_patch(image, box, self.input_size)
            if patch is None:
                continue
            patch = patch[:, :, ::-1]  # BGR -> RGB
            patch = Image.fromarray(patch.astype(np.uint8))
            patches.append(self.transform(patch))
            valid_idx.append(i)
        return patches, valid_idx

    def encode(self, image, boxes):
        boxes = np.asarray(boxes, dtype=np.float64)
        if len(boxes) == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        features = np.zeros((len(boxes), self.feature_dim), dtype=np.float32)
        patches, valid_idx = self._preprocess_patches(image, boxes)
        if not patches:
            return features

        tensor = torch.stack(patches).to(self.device)
        with torch.no_grad():
            for start in range(0, len(tensor), self.batch_size):
                batch = tensor[start:start + self.batch_size]
                emb = self.model(batch)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                features[valid_idx[start:start + len(batch)]] = emb.cpu().numpy()
        return features
