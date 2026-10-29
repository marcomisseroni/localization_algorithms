"""
Error metrics shared by evaluate.py and the analysis scripts, so that every report uses the same definitions.
"""

import numpy as np


def error_stats(err: np.ndarray) -> dict[str, float]:
    """
    Summary statistics of the localization errors [m] of one run.
    Returns NaN for every statistic if there are no errors.
    """
    if(err.size == 0):
        return {"n": 0, "mean": np.nan, "median": np.nan, "rmse": np.nan, "p90": np.nan, "p95": np.nan,
                "max": np.nan, "below_10cm": np.nan, "below_50cm": np.nan}
    return {
        "n": int(err.size),
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "p90": float(np.percentile(err, 90)),
        "p95": float(np.percentile(err, 95)),
        "max": float(err.max()),
        "below_10cm": float(np.mean(err < 0.10) * 100),     # [%]
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
