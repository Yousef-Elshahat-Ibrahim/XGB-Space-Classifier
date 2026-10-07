"""preprocess.py - raw broker detections -> per-band arrays, plus Milky Way dust (E(B-V), A_lambda)."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import sfdmap
from astropy import units as u
from dust_extinction.parameter_averages import F99

np.int = int  # sfdmap compatibility shim

SFD_MAP_PATH = Path(__file__).resolve().parent / "sfddata-master"
# Fixed band order (it matches how the training table was built).
BAND_ORDER = ("z", "r", "y", "i", "g", "u")
FILTERS_EFF = {"u": 3641, "g": 4704, "r": 6155, "i": 7504, "z": 8695, "y": 10056}   # Angstrom
QUALITY_FLAGS = ["psfFlux_flag", "pixelFlags_bad", "pixelFlags_saturated"]


@lru_cache(maxsize=1)
def _sfd():
    return sfdmap.SFDMap(str(SFD_MAP_PATH))


def get_ebv(ra, dec):
    return float(_sfd().ebv(ra, dec))


def extinction(ebv, bands, Rv=3.1):
    """F99 extinction A_lambda (mag) for each band letter."""
    wl = np.array([FILTERS_EFF[b] * 1e-4 for b in bands]) * u.micron
    return dict(zip(bands, (F99(Rv=Rv)(wl) * Rv * ebv).astype(float)))


def prepare_light_curve(df):
    """Clean the detections and split them per band.

    Returns (bands, ra, dec, span) where bands = {band: (t, flux, flux_err)}: arrays sorted by MJD,
    one point per MJD, unobserved/zero fluxes dropped, flux in uJy. `span` is last-minus-first MJD
    over all clean detections. Flux has the training-time +100 offset already added.
    """
    df = df[df[QUALITY_FLAGS].eq(False).all(axis=1)]
    if df.empty:
        raise ValueError("no usable detections left after quality cuts")

    bands = {}
    for b in BAND_ORDER:
        d = df[df["band_name"] == b]
        t = d["mjd"].to_numpy(float)
        f = d["psfFlux"].to_numpy(float) * 1e-3          # nJy -> uJy
        e = d["psfFluxErr"].to_numpy(float) * 1e-3
        keep = (f != 0) & ~np.isnan(f)
        t, f, e = t[keep], f[keep], e[keep]
        if len(t):
            t, first = np.unique(t, return_index=True)    # sort by time, keep first duplicate
            bands[b] = (t, f[first] + 100, e[first])

    span = float(df["mjd"].max() - df["mjd"].min())
    return bands, df["ra"].iloc[0], df["dec"].iloc[0], span
