from __future__ import annotations


def load_dataset(name: str, root: str, train: bool = True):
    """Dataset loading hook.

    Exact transforms, validation split, GTSRB preprocessing, augmentation, and
    caching are intentionally deferred until they are matched to the experiments
    reported in the manuscript.
    """
    raise NotImplementedError(
        f"Dataset pipeline for '{name}' must be aligned with the reported experiment settings before release."
    )
