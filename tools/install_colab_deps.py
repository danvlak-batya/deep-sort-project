"""Install dependencies for Google Colab."""
from __future__ import print_function

import subprocess
import sys


def pip_install(*packages):
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + list(packages)
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    # Core packages — no torchreid/trackeval (they break on Colab Python 3.12)
    pip_install(
        "numpy>=1.22,<2.1",
        "opencv-python",
        "scipy",
        "filterpy",
        "PyYAML",
        "scikit-learn",
        "Pillow",
        "tqdm",
        "ultralytics",
        "timm",
    )
    # Optional: RT-DETR detector only
    try:
        pip_install("transformers")
    except subprocess.CalledProcessError:
        print("Warning: transformers not installed (RT-DETR detector unavailable)")

    import ultralytics  # noqa: F401
    import timm  # noqa: F401
    import torch
    print("OK: ultralytics, timm, torch", torch.__version__)


if __name__ == "__main__":
    main()
