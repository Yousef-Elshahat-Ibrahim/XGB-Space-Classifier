import os
import glob
import traceback

import streamlit as st

from pipeline import classify_object
from model import load_classifier

st.set_page_config(page_title="Transient Classifier", page_icon="✨", layout="centered")

st.title("Transient Event Classifier")
st.caption(
    "Classifies LSST transient candidates (fetched via ALeRCE / Fink) using a Keras "
    "model trained on longer-baseline light curves. Because LSST currently only has "
    "a few months of survey history, accuracy on real LSST objects is still limited "
    "compared to the training data — this is a live demo of the deployment pipeline, "
    "not a production-accuracy classifier yet."
)

# ---- discover available models in the models/ folder ----
MODEL_DIR = "models"


@st.cache_resource(show_spinner=False)
def get_model(path):
    return load_classifier(path)


model_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.keras")))
model_names = [os.path.basename(m) for m in model_files]

if not model_names:
    st.error(f"No .keras model files found in '{MODEL_DIR}/'. Check the repo contents.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    broker = st.selectbox("Broker", ["alerce", "fink"])
with col2:
    model_choice = st.selectbox("Model", model_names)

# a few known example IDs to make the demo easy to run without hunting for a valid oid
EXAMPLE_IDS = {
    "— pick an example —": "",
    "Example 1": "170028510512414824",
    "Example 2": "313893023122980956",
    "Example 3": "170028516436869257",
}
example_choice = st.selectbox("Quick examples (optional)", list(EXAMPLE_IDS.keys()))
default_oid = EXAMPLE_IDS[example_choice]

oid = st.text_input("Object ID", value=default_oid, placeholder="e.g. 170028510512414824")

run = st.button("Classify", type="primary", disabled=not oid)

if run and oid:
    model_path = os.path.join(MODEL_DIR, model_choice)
    with st.spinner("Fetching light curve and running classification..."):
        try:
            model = get_model(model_path)
            result = classify_object(oid.strip(), model, broker=broker)
            score = float(result[0][0]) if hasattr(result, "__len__") else float(result)

            st.success("Classification complete")
            st.metric("Raw model output", f"{score:.4f}")
            st.progress(min(max(score, 0.0), 1.0))

            st.caption(
                "Score is the raw sigmoid output of the selected model, not a calibrated "
                "probability. Interpret direction/threshold according to how this model "
                "was trained."
            )
        except Exception as e:
            st.error("Something went wrong while classifying this object.")
            st.caption(
                f"({type(e).__name__}: {e}) — if this is a FileNotFoundError pointing "
                "at sfddata-master, the dust-map files likely didn't upload correctly, "
                "not that the object is missing."
            )
            with st.expander("Full traceback"):
                st.code(traceback.format_exc())

st.divider()
st.caption(
    "Known limitation: models trained on multi-year light curves currently see degraded "
    "accuracy on LSST's shorter observation baseline. A retrain on truncated light curves "
    "is in progress."
)