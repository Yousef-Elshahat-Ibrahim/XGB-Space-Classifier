"""
xgb_predict.py - give an object id, get the XGBoost prediction.

    from xgb_predict import predict_object
    predict_object("170028485725126711")
    # {'object_id': ..., 'label': 'TDE', 'probabilities': {'AGN': 0.02, 'SN': 0.05, 'TDE': 0.93}}
"""
from functools import lru_cache

import joblib
import numpy as np
from xgboost import XGBClassifier

from features import build_features
from fetch import get_lightcurve
from preprocess import get_ebv, prepare_light_curve

# ---- edit these paths ----------------------------------------------------------
MODEL_PATH = "models/XGB_model.json"
ENCODER_PATH = "models/label_encoder.pkl"      # LabelEncoder (or a list of class names)
# --------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load():
    """Load once: (model, feature columns stored in the booster, class names)."""
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    cols = model.get_booster().feature_names
    if not cols:
        raise ValueError("model has no stored feature names")
    enc = joblib.load(ENCODER_PATH)
    classes = [str(c) for c in (enc.classes_ if hasattr(enc, "classes_") else enc)]
    return model, list(cols), classes


def predict_object(oid, broker="alerce"):
    model, cols, classes = _load()
    bands, ra, dec, span = prepare_light_curve(get_lightcurve(str(oid), broker))
    X = build_features(bands, get_ebv(ra, dec), span, cols)
    proba = model.predict_proba(X, validate_features=False)[0]
    return {"object_id": str(oid),
            "label": classes[int(np.argmax(proba))],
            "probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)}}
