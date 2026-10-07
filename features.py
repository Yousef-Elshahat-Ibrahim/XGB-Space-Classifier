"""features.py - build the XGBoost feature vector for one live object.

Two feature groups, both identical to training:
  * legacy per-band statistics  (columns 'Filter_<b>_flux_*')
  * the tde_features families   (bazin_, potica_, shape_, ...)  + 'dt' and 'EBV'
Only the groups the model's columns actually need are computed.
"""
import numpy as np
from scipy.stats import kurtosis, median_abs_deviation, skew

import tde_features as tdf
from preprocess import extinction


def _slope(t, x):
    return (x[-1] - x[0]) / (t[-1] - t[0]) if t[-1] != t[0] else np.nan


def _legacy_band(prefix, t, x):
    """Statistics of one band. t sorted ascending, x = flux (+100 offset)."""
    q05, q25, q75, q95 = np.percentile(x, [5, 25, 75, 95])
    peak = int(np.argmax(x))
    median = np.median(x)
    dt = np.diff(t)
    denom = t[-1] - t[peak]
    res = {
        "min": x.min(), "max": x.max(), "mean": x.mean(),
        "std": np.std(x, ddof=1), "var": np.var(x, ddof=1), "median": median,
        "skew": skew(x, bias=False), "kurtosis": kurtosis(x, bias=False), "n_obs": len(x),
        "q05": q05, "q25": q25, "q75": q75, "q95": q95, "iqr": q75 - q25,
        "mad": median_abs_deviation(x, scale="normal"),
        "max_change": np.max(np.abs(np.diff(x) / dt)),
        "rise_time": t[peak] - t[0], "decay_time": denom,
        "slope_pre": _slope(t[:peak + 1], x[:peak + 1]),
        "slope_post": _slope(t[peak:], x[peak:]),
        "rise_decay_ratio": (t[peak] - t[0]) / denom if denom > 0 else np.nan,
        "time_above_median": np.sum(dt * (x[:-1] > median)),
        "duty_cycle": np.mean(x > median),
        "post_peak_fraction": x[-1] / x.max() if x.max() != 0 else np.nan,
        "num_peaks": np.sum((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])),
        "plaw_alpha": np.nan, "plaw_r2": np.nan,
    }
    t_post, x_post = t[peak + 1:] - t[peak], x[peak + 1:]      # power-law decay after the peak
    if len(x_post) >= 3 and np.all(x_post > 0):
        log_t, log_x = np.log(t_post), np.log(x_post)
        res["plaw_alpha"] = -np.polyfit(log_t, log_x, 1)[0]
        res["plaw_r2"] = np.corrcoef(log_t, log_x)[0, 1] ** 2
    return {f"{prefix}_{k}": v for k, v in res.items()}


def _legacy_features(bands):
    feats = {}
    for b, (t, f, _) in bands.items():
        if len(t) > 3:                                   # fewer points -> features stay NaN
            feats.update(_legacy_band(f"Filter_{b}_flux", t, f))
    return feats


def build_features(bands, ebv, span, cols):
    """Feature vector (1, len(cols)) in the model's column order; unknown/uncomputable -> NaN."""
    row = {"dt": span, "EBV": ebv}
    if any(c.startswith("Filter_") for c in cols):
        row.update(_legacy_features(bands))
    families = tdf.families_for(cols)
    if families:
        row.update(tdf.extract_object_features(bands, extinction(ebv, list(bands)), families))
    return np.array([[row.get(c, np.nan) for c in cols]], dtype=float)
