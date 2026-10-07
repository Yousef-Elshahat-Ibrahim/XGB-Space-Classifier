"""dev_tools.py - helpers for building test sets. Not used by predict_object."""
import io

import pandas as pd
import requests

from fetch import FINK_URL, alerce_client


def search_fink_objects(tag="extragalactic_new_candidate", n_candidates=300, min_detections=30, top_n=10):
    """Fink object ids with a given tag, keeping the best-observed ones."""
    r = requests.post(f"{FINK_URL}/tags", timeout=60,
                      json={"tag": tag, "columns": "r:diaObjectId", "n": str(n_candidates)})
    ids = pd.read_json(io.BytesIO(r.content))["r:diaObjectId"].astype(str).unique().tolist()
    if not ids:
        return []

    r = requests.post(f"{FINK_URL}/sources", timeout=120,
                      json={"diaObjectId": ",".join(ids), "columns": "r:diaObjectId",
                            "output-format": "json"})
    counts = pd.read_json(io.BytesIO(r.content))["r:diaObjectId"].astype(str).value_counts()
    return counts[counts >= min_detections].sort_values(ascending=False).head(top_n).index.tolist()


def search_objects(survey="lsst", classifier="stamp_classifier_rubin_beta_20260421",
                   class_name="AGN", probability=0.89, n_det=40, page_size=100):
    """ALeRCE object ids of a given stamp-classifier class."""
    res = alerce_client().query_objects(survey=survey, classifier=classifier, class_name=class_name,
                                        probability=probability, n_det=n_det, page_size=page_size,
                                        format="pandas")
    return res["oid"].tolist()
