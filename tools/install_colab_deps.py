"""Install dependencies for Google Colab."""
from __future__ import print_function

import subprocess
import sys


def pip_install(*packages, **kwargs):
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if kwargs.get("force_reinstall"):
        cmd.append("--force-reinstall")
    cmd.extend(packages)
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    # numpy+scipy must be installed as a matched pair on Colab Python 3.12.
    # Pinning numpy<2 breaks scipy/trackeval (ImportError: _center).
    pip_install("numpy>=2.0,<2.3", "scipy>=1.14", force_reinstall=True)

    pip_install(
        "opencv-python",
        "filterpy",
        "PyYAML",
        "scikit-learn",
        "Pillow",
        "tqdm",
        "ultralytics",
        "timm",
        "trackeval",
    )

    try:
        pip_install("transformers")
    except subprocess.CalledProcessError:
        print("Warning: transformers not installed (RT-DETR detector unavailable)")

    import numpy
    import scipy
    import scipy.linalg  # noqa: F401
    import trackeval  # noqa: F401
    import ultralytics  # noqa: F401
    import timm  # noqa: F401
    import torch

    print("\n=== Dependencies OK ===")
    print("numpy:  ", numpy.__version__)
    print("scipy:  ", scipy.__version__)
    print("torch:  ", torch.__version__)
    print("timm:   ", timm.__version__)
    print("trackeval: OK")


if __name__ == "__main__":
    main()
