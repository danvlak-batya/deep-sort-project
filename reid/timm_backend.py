"""Person ReID via timm (Colab-compatible, no torchreid install needed)."""
import numpy as np
import torch
import timm
import torchvision.transforms as T
from PIL import Image

from reid.base import BaseReID
from utils.image import extract_image_patch

def _first_timm_model(patterns, fallbacks=None):
    """Find first available pretrained timm model by glob patterns."""
    for pattern in patterns:
        matches = timm.list_models(pattern, pretrained=True)
        if matches:
            return matches[0]
    for name in (fallbacks or []):
        matches = timm.list_models(name, pretrained=True)
        if matches:
            return matches[0]
        if not name.endswith("*"):
            matches = timm.list_models(name + "*", pretrained=True)
            if matches:
                return matches[0]
    return None


def _resolve_timm_name(model_key):
    """Map registry keys to an available timm model (OSNet may be absent in some timm versions)."""
    search = {
        "osnet_x0_25": (
            ["osnet_x0_25*", "osnet_x0_5*", "osnet*"],
            ["mobilenetv3_small_100*", "resnet18*"],
        ),
        "resnet50_ibn": (
            ["resnet50_ibn*", "resnet50*"],
            ["resnet34*"],
        ),
        "fastreid_sbs": (
            ["osnet_x1_0*", "osnet_x0_75*", "resnet101*"],
            ["efficientnet_b0*", "resnet50*"],
        ),
    }

    if model_key in search:
        patterns, fallbacks = search[model_key]
        name = _first_timm_model(patterns, fallbacks)
        if name:
            if name not in patterns[0].replace("*", ""):
                print("ReID %s -> using timm model: %s" % (model_key, name))
            return name

    name = _first_timm_model([model_key, model_key + "*"], ["resnet18*"])
    if name:
        return name
    raise ValueError(
        "No timm model found for %s. Try: pip install -U timm" % model_key)


class TimmReIDEncoder(BaseReID):
    """ReID encoder using timm pretrained backbones (OSNet, ResNet-IBN, etc.)."""

    def __init__(self, model_name="osnet_x0_25", batch_size=32, device=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        timm_name = _resolve_timm_name(model_name)
        self.timm_name = timm_name

        self.model = timm.create_model(timm_name, pretrained=True, num_classes=0)
        self.model.eval()
        self.model.to(self.device)
        self.input_size = (256, 128)
        self.transform = T.Compose([
            T.Resize(self.input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.input_size[0], self.input_size[1], device=self.device)
            self._feature_dim = int(self.model(dummy).shape[-1])

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
            patch = patch[:, :, ::-1]
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
                if emb.dim() > 2:
                    emb = emb.view(emb.size(0), -1)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                features[valid_idx[start:start + len(batch)]] = emb.cpu().numpy()
        return features
