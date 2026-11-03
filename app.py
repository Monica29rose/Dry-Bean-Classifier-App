"""Streamlit front end. It only LOADS the saved pipelines from model/ and scores the uploaded
test CSV - nothing is ever fitted here (assignment requirement).

Run locally:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from eval_utils import METRIC_NAMES, compute_metrics, confusion_array, report_frame

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_CSV = BASE_DIR / "test_data.csv"

APP_TITLE = "Dry Bean Variety Classifier - Model Evaluation Lab"
APP_BLURB = ("Seven registered dry-bean varieties are identified from 16 shape and size measurements "
             "extracted from grain images. Pick one of five pre-trained classifiers, score a test "
             "file and inspect accuracy, AUC, precision, recall, F1, MCC and the confusion matrix. "
             "Models are loaded from disk and are never retrained here.")
DATASET_NOTE = ("**Dataset:** Dry Bean Dataset, UCI Machine Learning Repository (13,611 grains, "
                "16 features, 7 classes: Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira).  \n"
                "**Citation:** Koklu, M. and Ozkan, I.A. (2020). Multiclass classification of dry "
                "beans using computer vision and machine learning techniques. Computers and "
                "Electronics in Agriculture, 174, 105507.")

st.set_page_config(page_title=APP_TITLE, layout="wide")


@st.cache_resource
def load_meta():
    return json.loads((MODEL_DIR / "metadata.json").read_text())


@st.cache_resource
def load_model(file_name):
    return joblib.load(MODEL_DIR / file_name)


def score(model_name, X, y_true, meta):
    """Predict with a saved pipeline and compute all metrics (no fitting)."""
    pipe = load_model(meta["models"][model_name])
    classes = [str(c) for c in pipe.classes_]
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
    return pred, proba, classes, compute_metrics(y_true, pred, proba, classes,
                                                 meta.get("positive_class"))


def safe_score(model_name, X, y_true, meta):
    """score() with a readable error instead of a stack trace (e.g. wrong column types)."""
    try:
        return score(model_name, X, y_true, meta)
    except Exception as exc:
        st.error(f"The saved model could not score this file ({type(exc).__name__}: {exc}). "
                 "Check that numeric columns contain numbers and the columns match the training data.")
        st.stop()


def draw_confusion(cm, classes, normalise):
    data = cm.astype(float)
    if normalise:
        data = data / np.maximum(data.sum(axis=1, keepdims=True), 1)
    n = len(classes)
    size = max(4.0, 0.6 * n + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalise else " (counts)"))
    for i in range(n):
        for j in range(n):
            txt = f"{data[i, j]:.2f}" if normalise else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if data[i, j] > data.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def fmt(x):
    return "n/a" if x != x else f"{x:.4f}"      # NaN-safe (AUC can be undefined)


meta = load_meta()
st.title(APP_TITLE)
st.caption(APP_BLURB)

# ------------------------------- sidebar: model + upload --------------------------------
with st.sidebar:
    st.header("1. Model")
    model_name = st.selectbox("Choose a classifier", list(meta["models"].keys()))
    st.header("2. Test data")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    if SAMPLE_CSV.exists():
        st.download_button("Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")
    with st.expander("About the dataset"):
        st.markdown(DATASET_NOTE)
    with st.expander("Required columns"):
        st.write(f"Target column: **{meta['target']}**")
        st.write(meta["feature_columns"])

# An uploaded file takes precedence; otherwise the bundled hold-out set is used so that results
# are visible as soon as the app opens (the upload option is still fully functional).
if uploaded is not None:
    source, source_name = uploaded, getattr(uploaded, "name", "uploaded file")
elif SAMPLE_CSV.exists():
    source, source_name = SAMPLE_CSV, "bundled test_data.csv (nothing uploaded yet)"
else:
    st.info("Upload a test CSV in the sidebar to begin.")
    st.stop()
st.info(f"Data source: {source_name}")

# ------------------------------- read + validate ----------------------------------------
try:
    df = pd.read_csv(source)
except Exception as exc:                           # empty file, wrong encoding, not a CSV ...
    st.error(f"Could not read the file as CSV: {exc}")
    st.stop()

target, features = meta["target"], meta["feature_columns"]
missing = [c for c in features + [target] if c not in df.columns]
if missing:
    st.error(f"The uploaded file is missing required column(s): {missing}")
    st.stop()
extra = [c for c in df.columns if c not in features + [target]]
if extra:
    st.warning(f"Ignoring columns that the model does not use: {extra}")
df = df.dropna(subset=[target]).reset_index(drop=True)
if df.empty:
    st.error("No rows with a target value were found.")
    st.stop()

X = df[features]                                    # same order as at training time
y_true = df[target].astype(str).values
unknown = sorted(set(y_true) - set(meta["classes"]))
if unknown:
    st.error(f"Target contains labels the models were not trained on: {unknown}. "
             f"Expected: {meta['classes']}")
    st.stop()

pred, proba, classes, metrics = safe_score(model_name, X, y_true, meta)
st.success(f"Scored {len(df)} rows with **{model_name}** (pre-trained, no retraining).")

tab1, tab2, tab3, tab4 = st.tabs(["Metrics", "Confusion matrix & report",
                                  "Compare all models", "Predictions"])

with tab1:
    st.subheader(f"Evaluation metrics - {model_name}")
    row1, row2 = st.columns(3), st.columns(3)
    for col, name in zip(row1 + row2, METRIC_NAMES):
        col.metric(name, fmt(metrics[name]))
    st.subheader("Class distribution in this file")
    st.bar_chart(pd.Series(y_true).value_counts())
    avg = meta["averaging"]
    st.caption("Precision/Recall/F1 are for the positive class '" + str(meta["positive_class"]) + "'."
               if avg == "binary" else
               "Multi-class: Precision/Recall/F1 are macro-averaged; AUC is one-vs-rest macro.")

with tab2:
    cm = confusion_array(y_true, pred, classes)
    normalise = st.checkbox("Row-normalise the matrix", value=False)
    left, right = st.columns([1, 1])
    with left:
        fig = draw_confusion(cm, classes, normalise)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Classification report")
        st.dataframe(report_frame(y_true, pred, classes))

with tab3:
    st.subheader("All five models on the uploaded file")
    rows = []
    for name in meta["models"]:
        m = safe_score(name, X, y_true, meta)[3]
        rows.append({"ML Model Name": name, **{k: round(m[k], 4) for k in METRIC_NAMES}})
    table = pd.DataFrame(rows)
    st.dataframe(table)
    best = table.loc[table["MCC"].idxmax(), "ML Model Name"]
    st.info(f"Highest MCC on this file: **{best}**")

with tab4:
    out = df.copy()
    out["predicted_" + target] = pred
    for i, c in enumerate(classes):
        out[f"proba_{c}"] = proba[:, i].round(4)
    st.dataframe(out.head(200))
    st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode("utf-8"),
                       file_name="predictions.csv", mime="text/csv")
