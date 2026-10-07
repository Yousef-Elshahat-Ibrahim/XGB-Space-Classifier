"""fetch.py - download an object's light curve from a broker, with a local CSV cache."""
import io
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path("events_data")
FINK_URL = "https://api.lsst.fink-portal.org/api/v1"
FINK_COLUMNS = ("r:diaObjectId,r:midpointMjdTai,r:band,r:psfFlux,r:psfFluxErr,r:ra,r:dec,"
                "r:psfFlux_flag,r:pixelFlags_bad,r:pixelFlags_saturated")
FINK_RENAME = {"diaObjectId": "oid", "midpointMjdTai": "mjd", "band": "band_name"}


@lru_cache(maxsize=1)
def alerce_client():
    from alerce.core import Alerce          # imported lazily: only needed for ALeRCE
    return Alerce()


def _fetch_alerce(oid):
    return pd.DataFrame(alerce_client().query_lightcurve(oid, survey="lsst")["detections"])


def _fetch_fink(oid):
    r = requests.post(f"{FINK_URL}/sources", timeout=60,
                      json={"diaObjectId": oid, "columns": FINK_COLUMNS, "output-format": "json"})
    r.raise_for_status()
    df = pd.read_json(io.BytesIO(r.content))
    return df.rename(columns=lambda c: c.replace("r:", "")).rename(columns=FINK_RENAME)


_FETCHERS = {"alerce": _fetch_alerce, "fink": _fetch_fink}


def get_lightcurve(oid, broker):
    """Return the raw detections table; downloaded once, then read from events_data/."""
    if broker not in _FETCHERS:
        raise ValueError(f"Unknown broker '{broker}'. Choose from: {list(_FETCHERS)}")
    path = DATA_DIR / f"{oid}_{broker}.csv"
    if path.exists():
        return pd.read_csv(path)
    df = _FETCHERS[broker](oid)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    return df
