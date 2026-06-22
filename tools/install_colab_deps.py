"""Install dependencies for Google Colab."""
from __future__ import print_function

import subprocess
import sys


def pip_install(*args):
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + list(args)
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    pip_install(
        "numpy>=1.22,<2.1", "opencv-python", "scipy", "filterpy",
        "PyYAML", "scikit-learn", "Pillow", "tqdm",
        "ultralytics", "transformers", "trackeval")
    pip_install("git+https://github.com/KaiyangZhou/deep-person-reid.git")

    import ultralytics  # noqa: F401
    import torchreid  # noqa: F401
    import torch
    print("OK: ultralytics, torchreid, torch", torch.__version__)


if __name__ == "__main__":
    main()
