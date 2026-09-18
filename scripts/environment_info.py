from __future__ import annotations

import json
import platform
import sys
from importlib import metadata

import torch


def package_version(distribution: str):
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def main():
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "default_dtype": str(torch.get_default_dtype()),
        "packages": {
            "numpy": package_version("numpy"),
            "torch": package_version("torch"),
            "torchvision": package_version("torchvision"),
            "PyYAML": package_version("PyYAML"),
            "Pillow": package_version("Pillow"),
            "scipy": package_version("scipy"),
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
