from reid.base import BaseReID
from reid.torchreid_backend import TorchReIDEncoder
from reid.fastreid_backend import FastReIDEncoder

REID_REGISTRY = {
    "osnet_x0_25": lambda **kw: TorchReIDEncoder(model_name="osnet_x0_25", **kw),
    "resnet50_ibn": lambda **kw: TorchReIDEncoder(model_name="resnet50_ibn_a", **kw),
    "fastreid_sbs": lambda **kw: FastReIDEncoder(**kw),
}


def create_reid(name, **kwargs):
    if name not in REID_REGISTRY:
        raise ValueError(
            "Unknown ReID '%s'. Choose from: %s" % (name, ", ".join(REID_REGISTRY)))
    return REID_REGISTRY[name](**kwargs)


def list_reid_models():
    return list(REID_REGISTRY.keys())
