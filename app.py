import traceback

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from xgb_predict import predict_object

st.set_page_config(page_title="Transient Classifier", page_icon="✨", layout="centered")

GROUPS = ["AGN", "SN", "TDE"]
COLORS = {"AGN": "#e8743b", "SN": "#3b8ee8", "TDE": "#8e44ad", "Other": "#999999"}
EXAMPLES = {
    "— pick an example —": "",
    "Example 1": "314051320305156133",
    "Example 2": "170028510512414824",
    "Example 3": "313893023122980956",
    "Example 4": "170028516436869257",
}


def group_probs(probs):
    """Merge every supernova subtype (SN*, SLSN*) into one 'SN' class."""
    out = {}
    for name, p in probs.items():
        key = "SN" if name.startswith(("SN", "SLSN")) else name
        out[key] = out.get(key, 0.0) + p
    return out


@st.cache_data(show_spinner=False)
def run(oid):
    return predict_object(oid, broker="alerce")


def ternary_plot(g):
    fig = go.Figure(go.Scatterternary(
        a=[g["AGN"]], b=[g["SN"]], c=[g["TDE"]], mode="markers",
        marker=dict(size=16, color="#d62728", line=dict(width=2, color="white")),
        hovertemplate="AGN %{a:.1%}<br>SN %{b:.1%}<br>TDE %{c:.1%}<extra></extra>"))
    fig.update_layout(
        ternary=dict(sum=1,
                     aaxis=dict(title="AGN", min=0, tickformat=".0%"),
                     baxis=dict(title="SN", min=0, tickformat=".0%"),
                     caxis=dict(title="TDE", min=0, tickformat=".0%")),
        margin=dict(l=40, r=40, t=20, b=20), height=420, showlegend=False)
    return fig


st.title("Transient Event Classifier")
st.caption(
    "Classifies LSST transient candidates (fetched via ALeRCE) with an XGBoost model. "
    "LSST only has a few months of history so far, so accuracy on real objects is still "
    "limited — this is a live demo of the pipeline, not a production-accuracy classifier."
)

example = st.selectbox("Quick examples (optional)", list(EXAMPLES))
oid = st.text_input("Object ID", value=EXAMPLES[example], placeholder="e.g. 314051320305156133").strip()

if st.button("Classify", type="primary", disabled=not oid):
    with st.spinner("Fetching light curve and running classification..."):
        try:
            res = run(oid)
        except Exception as e:
            st.error("Something went wrong while classifying this object.")
            st.caption(f"({type(e).__name__}: {e}) — if this is a FileNotFoundError pointing at "
                       "sfddata-master, the dust-map files are missing, not the object.")
            with st.expander("Full traceback"):
                st.code(traceback.format_exc())
            st.stop()

    g = group_probs(res["probabilities"])
    top = max(g, key=g.get)

    c1, c2 = st.columns(2)
    c1.metric("Most likely class", top)
    c2.metric("Probability", f"{g[top]:.1%}")

    if all(k in g for k in GROUPS):
        st.plotly_chart(ternary_plot(g), use_container_width=True)
        st.caption("Corners are pure classes; the closer the dot is to a corner, the more "
                   "confident the model is. A dot near the centre means the model can't separate them.")

    df = pd.DataFrame({"class": list(g), "probability": list(g.values())}).sort_values("probability")
    bar = go.Figure(go.Bar(x=df["probability"], y=df["class"], orientation="h",
                           marker_color=[COLORS.get(c, COLORS["Other"]) for c in df["class"]],
                           text=[f"{p:.1%}" for p in df["probability"]], textposition="outside"))
    bar.update_layout(xaxis=dict(range=[0, 1], tickformat=".0%"), height=200,
                      margin=dict(l=10, r=30, t=10, b=10))
    st.plotly_chart(bar, use_container_width=True)

    with st.expander("All model classes (before grouping)"):
        full = pd.Series(res["probabilities"], name="probability").sort_values(ascending=False)
        st.dataframe(full.map("{:.2%}".format))

st.divider()
st.caption("Known limitation: LSST's observation baseline is still short compared with the "
           "training light curves, so predictions on very young objects can be unreliable.")
