from reid.base import BaseReID
from reid.timm_backend import TimmReIDEncoder

REID_REGISTRY = {
    "osnet_x0_25": lambda **kw: TimmReIDEncoder(model_name="osnet_x0_25", **kw),
    "resnet50_ibn": lambda **kw: TimmReIDEncoder(model_name="resnet50_ibn", **kw),
    "fastreid_sbs": lambda **kw: TimmReIDEncoder(model_name="fastreid_sbs", **kw),
}


def create_reid(name, **kwargs):
    if name not in REID_REGISTRY:
        raise ValueError(
            "Unknown ReID '%s'. Choose from: %s" % (name, ", ".join(REID_REGISTRY)))
    return REID_REGISTRY[name](**kwargs)


def list_reid_models():
    return list(REID_REGISTRY.keys())
