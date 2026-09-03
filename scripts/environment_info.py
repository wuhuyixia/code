from __future__ import annotations

import json
import platform
import sys

import torch


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
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
