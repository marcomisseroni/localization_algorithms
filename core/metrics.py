"""
Error metrics shared by evaluate.py and the analysis scripts, so that every report uses the same definitions.
"""

import numpy as np


def error_stats(err: np.ndarray) -> dict[str, float]:
    """
    Summary statistics of the localization errors [m] of one run.
    Returns NaN for every statistic if there are no errors.
    """
    keys = ["mean", "median", "rmse", "std", "min", "max", "p25", "p75", "p90", "p95",
            "below_5cm", "below_10cm", "below_25cm", "below_50cm"]
    if(err.size == 0):
        return {"n": 0, **{k: np.nan for k in keys}}
    return {
        "n": int(err.size),
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "std": float(err.std()),
        "min": float(err.min()),
        "max": float(err.max()),
        "p25": float(np.percentile(err, 25)),
        "p75": float(np.percentile(err, 75)),
        "p90": float(np.percentile(err, 90)),
        "p95": float(np.percentile(err, 95)),
        "below_5cm": float(np.mean(err < 0.05) * 100),      # [%]
        "below_10cm": float(np.mean(err < 0.10) * 100),     # [%]
        "below_25cm": float(np.mean(err < 0.25) * 100),     # [%]
        "below_50cm": float(np.mean(err < 0.50) * 100),     # [%]
    }


def binned_median(x: np.ndarray, y: np.ndarray, bin_width: float) -> tuple[np.ndarray, np.ndarray]:
    """Median of y in consecutive bins of x of width bin_width. Returns the bin centres and the medians (empty bins skipped)."""
    edges = np.arange(0, np.ceil(x.max() / bin_width) * bin_width + bin_width, bin_width)
    centres, medians = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (x >= lo) & (x < hi)
        if(k.any()):
            centres.append((lo + hi) / 2)
            medians.append(np.median(y[k]))
    return np.array(centres), np.array(medians)
