import re
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix

st.set_page_config(page_title="Medical Abstract Segmentation", page_icon="🩺", layout="wide")

# ---------- LOAD ARTIFACTS ----------
@st.cache_resource
def load_artifacts():
    clf = joblib.load("rct_model.joblib")
    tfidf = joblib.load("rct_tfidf.joblib")
    le = joblib.load("rct_label_encoder.joblib")
    metrics = joblib.load("rct_metrics.joblib")
    return clf, tfidf, le, metrics


clf, tfidf, le, metrics = load_artifacts()

COLORS = {
    "BACKGROUND": "#8ecae6",
    "OBJECTIVE": "#ffb703",
    "METHODS": "#fb8500",
    "RESULTS": "#219ebc",
    "CONCLUSIONS": "#2a9d8f",
}

EXAMPLE_ABSTRACT = (
    "This study aimed to evaluate the impact of a 12-week structured exercise "
    "program on markers of systemic inflammation in adults with type 2 diabetes. "
    "A total of 84 participants were randomly assigned to either a supervised "
    "exercise group or a standard-care control group. Blood samples were "
    "collected at baseline and at 12 weeks to measure C-reactive protein and "
    "interleukin-6 levels. Participants completed three 45-minute sessions per "
    "week combining aerobic and resistance training. The exercise group showed "
    "a significant reduction in C-reactive protein compared to controls "
    "( p < 0.01 ) . Interleukin-6 levels also decreased significantly in the "
    "intervention group but not in controls. No adverse events related to the "
    "exercise program were reported. These findings suggest that structured "
    "exercise can meaningfully reduce systemic inflammation in adults with "
    "type 2 diabetes and may support its use as an adjunct therapy."
)


def split_sentences(text):
    text = text.strip()
    # If the user pasted one sentence per line, respect that.
    if "\n" in text and len([l for l in text.split("\n") if l.strip()]) > 1:
        sentences = [l.strip() for l in text.split("\n") if l.strip()]
    else:
        # Basic regex sentence splitter (avoids splitting on common abbreviations/decimals)
        text = re.sub(r"(?<=[a-zA-Z]{2})\.(?=\d)", ".", text)  # no-op safeguard
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def predict_sentences(sentences):
    n = len(sentences)
    df = pd.DataFrame({
        "text": sentences,
        "line_number": range(n),
        "total_lines": [n] * n,
    })
    df["line_pct"] = df["line_number"] / max(n, 1)

    X_text = tfidf.transform(df["text"])
    X_num = csr_matrix(df[["line_number", "total_lines", "line_pct"]].values.astype(float))
    X = hstack([X_text, X_num]).tocsr()

    preds = clf.predict(X)
    probs = clf.predict_proba(X).max(axis=1)
    labels = le.inverse_transform(preds)
    return labels, probs


# ---------- HEADER ----------
st.title("🩺 Medical Abstract Segmentation")
st.markdown(
    "Classifies each sentence of a biomedical abstract into "
    "**Background / Objective / Methods / Results / Conclusions** — trained on the "
    "**PubMed 20k RCT** dataset (180k labeled sentences from randomized controlled trials)."
)
st.info(
    "The original project compared fine-tuned **ELECTRA** and **DeBERTa** transformers "
    "(~90%+ accuracy). This live demo uses a **TF-IDF + Logistic Regression** model instead — "
    "the transformer checkpoints weren't saved from the original training run, and re-training "
    "them needs a GPU this environment doesn't have. This version runs entirely in the browser "
    "session with no GPU required, at a modest accuracy trade-off.",
    icon="ℹ️",
)

col1, col2, col3 = st.columns(3)
col1.metric("Test Accuracy", f"{metrics['test_acc']*100:.1f}%")
col2.metric("Weighted F1", f"{metrics['test_f1']:.3f}")
col3.metric("Training sentences", f"{metrics['n_train']:,}")

st.divider()

# ---------- INPUT ----------
st.subheader("Try it — paste an abstract")
use_example = st.checkbox("Use example abstract", value=True)
text_input = st.text_area(
    "Abstract text (paste as one paragraph, or one sentence per line):",
    value=EXAMPLE_ABSTRACT if use_example else "",
    height=180,
)

segment_clicked = st.button("Segment Abstract", type="primary")

st.divider()

if segment_clicked and text_input.strip():
    sentences = split_sentences(text_input)
    labels, probs = predict_sentences(sentences)

    st.subheader("Segmented Output")
    for sent, label, prob in zip(sentences, labels, probs):
        color = COLORS.get(label, "#888")
        st.markdown(
            f"""<div style="border-left: 5px solid {color}; padding: 8px 14px; margin-bottom: 8px; background: rgba(128,128,128,0.07); border-radius: 4px;">
            <span style="background:{color}; color:white; padding:2px 8px; border-radius:10px; font-size:0.75em; font-weight:600;">{label}</span>
            <span style="color:#999; font-size:0.75em; margin-left:8px;">{prob*100:.0f}% confidence</span>
            <div style="margin-top:6px;">{sent}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Section Breakdown")
    label_df = pd.Series(labels).value_counts().reindex(
        ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"], fill_value=0
    )
    st.bar_chart(label_df)

st.caption(
    "Dataset: PubMed 20k RCT (Dernoncourt & Lee) · "
    "Stack: Python, scikit-learn, TF-IDF, Streamlit — built by Vyshnavi V"
)
