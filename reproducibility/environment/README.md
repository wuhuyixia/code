# Execution environment provenance

The manuscript reports the following execution environment:

- Windows 10
- NVIDIA GeForce RTX 3090 GPU (24 GB)
- Python 3.8.13
- PyTorch 2.4.0
- CUDA 12.1
- cuDNN 9.1.0

For the archival release, run:

```bash
python scripts/environment_info.py > reproducibility/environment/environment.json
```

on the original experiment machine (or its preserved environment). This records
the Python/platform/PyTorch/CUDA/cuDNN information together with installed
versions of the main Python dependencies. The generated JSON should be committed
with the release.

Do not replace unknown dependency versions with guessed values.
