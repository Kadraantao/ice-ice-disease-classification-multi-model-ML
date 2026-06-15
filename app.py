"""
Ice-Ice Disease Classification - Multi-Model Comparison Dashboard

Compares predictions from 4 classifiers (SVM, Random Forest, KNN, Logistic
Regression) across 4 preprocessing configurations (baseline, median, CLAHE,
median+CLAHE) - 16 model-configuration combinations in total.

Includes a One-Class SVM gate to reject non-seaweed inputs.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import json
import os
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="Ice-Ice Disease Classification Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Load configuration, all 16 classifiers, and OOD gate
# ============================================================
@st.cache_resource
def load_everything():
    """Load config, all 16 classifier pipelines, and OneClassSVM gate."""
    with open('models/config.json') as f:
        cfg = json.load(f)

    # Load 16 classifiers: nested dict {classifier_name: {config_name: pipeline}}
    models = {}
    for clf_name in cfg['classifiers']:
        safe_name = clf_name.lower().replace(' ', '_')
        models[clf_name] = {}
        for cfg_name in cfg['configs']:
            path = f'models/{safe_name}_{cfg_name}.joblib'
            if os.path.exists(path):
                models[clf_name][cfg_name] = joblib.load(path)

    # OOD gate (optional)
    oneclass_path = 'models/oneclass_seaweed.joblib'
    oneclass_gate = joblib.load(oneclass_path) if os.path.exists(oneclass_path) else None

    return cfg, models, oneclass_gate


try:
    CONFIG, MODELS, ONECLASS_GATE = load_everything()
except FileNotFoundError as e:
    st.error(f"Could not load models. Run the notebook end-to-end first to "
             f"generate the models/ folder.\n\nDetails: {e}")
    st.stop()


# ============================================================
# Constants from saved config
# ============================================================
CLASS_NAMES    = CONFIG['class_names']
IMG_SIZE       = CONFIG['img_size']
HIST_BINS      = CONFIG['hist_bins']
CLAHE_CLIP     = CONFIG['clahe_clip']
CLAHE_TILE     = tuple(CONFIG['clahe_tile'])
MEDIAN_KERNEL  = CONFIG['median_kernel']
CLASSIFIERS    = CONFIG['classifiers']
CONFIGS        = CONFIG['configs']
BEST_CLF       = CONFIG['best_classifier']
BEST_CFG       = CONFIG['best_config']
BEST_F1        = CONFIG['best_f1']
BEST_ACC       = CONFIG['best_accuracy']
ALL_RESULTS    = CONFIG['all_results']

CFG_LABELS = {
    'baseline': 'Baseline',
    'median':   'Median Filter',
    'clahe':    'CLAHE',
    'both':     'Median + CLAHE',
}

ABLATION_CONFIGS = {
    'baseline': (False, False),
    'median':   (True,  False),
    'clahe':    (False, True),
    'both':     (True,  True),
}


# ============================================================
# Preprocessing functions (must match notebook EXACTLY)
# ============================================================
def apply_median(img):
    return cv2.medianBlur(img, MEDIAN_KERNEL)


def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    l_enhanced = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)


def preprocess(img, use_median=False, use_clahe=False):
    if use_median:
        img = apply_median(img)
    if use_clahe:
        img = apply_clahe(img)
    return img


def extract_hsv_histogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    histograms = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [HIST_BINS], [0, 256])
        histograms.append(hist.flatten())
    feature = np.concatenate(histograms)
    feature = feature / (feature.sum() + 1e-7)
    return feature


# ============================================================
# UI - Header
# ============================================================
st.title("🌿 Ice-Ice Disease Classification Dashboard")
st.markdown(
    "**Multi-Model Comparison Tool** — Compare predictions from **4 classifiers** "
    "across **4 preprocessing configurations** (16 combinations total)."
)

# Top-line metrics row (KPI cards)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Classifiers", len(CLASSIFIERS))
with col2:
    st.metric("Configurations", len(CONFIGS))
with col3:
    st.metric("Total Combinations", len(CLASSIFIERS) * len(CONFIGS))
with col4:
    st.metric("Best F1-Score", f"{BEST_F1:.1f}%",
              delta=f"{BEST_CLF} + {CFG_LABELS[BEST_CFG]}")

st.markdown("---")


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("📊 Benchmark Results")
    st.markdown(
        f"**Best combination identified during training:**\n"
        f"- Classifier: **{BEST_CLF}**\n"
        f"- Preprocessing: **{CFG_LABELS[BEST_CFG]}**\n"
        f"- F1-Score: **{BEST_F1:.1f}%**\n"
        f"- Accuracy: **{BEST_ACC:.1f}%**"
    )

    st.markdown("---")
    st.subheader("🏆 Top 5 Combinations")
    top5 = sorted(ALL_RESULTS, key=lambda r: r['F1-Score'], reverse=True)[:5]
    top5_df = pd.DataFrame(top5)[['Classifier', 'Configuration', 'F1-Score']]
    top5_df.columns = ['Classifier', 'Config', 'F1 (%)']
    top5_df['F1 (%)'] = top5_df['F1 (%)'].apply(lambda x: f'{x:.1f}')
    st.dataframe(top5_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("🚪 OOD Gate")
    if ONECLASS_GATE is not None:
        st.success("✅ One-Class SVM enabled")
    else:
        st.warning("⚠️ One-Class SVM not loaded")

    st.markdown("---")
    st.caption(
        "**Pipeline**: Image → Preprocessing → HSV histogram (96-dim) → "
        "Classifier → Prediction"
    )


# ============================================================
# File uploader
# ============================================================
st.header("📤 Upload an Image")
uploaded_file = st.file_uploader(
    "Choose a seaweed image (JPG, JPEG, or PNG)",
    type=['jpg', 'jpeg', 'png'],
    help="Upload an image of Kappaphycus alvarezii seaweed"
)

if uploaded_file is None:
    st.info("👆 Upload an image to run all 16 model-configuration combinations.")
    st.stop()


# ============================================================
# Load and prepare image
# ============================================================
img_bytes = uploaded_file.read()
img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
img_np = np.array(img_pil)
img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))


# ============================================================
# OOD check: is this even a seaweed image?
# ============================================================
if ONECLASS_GATE is not None:
    baseline_features = extract_hsv_histogram(img_resized).reshape(1, -1)
    is_seaweed = ONECLASS_GATE.predict(baseline_features)[0] == 1

    if not is_seaweed:
        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            st.image(img_pil, caption=uploaded_file.name, use_container_width=True)
        st.error(
            "🚫 **This image does not appear to be a *Kappaphycus alvarezii* "
            "seaweed sample.** The One-Class SVM gate flagged it as "
            "out-of-distribution. Please upload a seaweed image."
        )
        with st.expander("Why is this happening?"):
            st.markdown(
                "All 4 classifiers in this dashboard are closed-set "
                "(trained only on `healthy` vs `early_signs`). To prevent "
                "meaningless predictions on non-seaweed inputs, a One-Class "
                "SVM gate learns the boundary of the training distribution "
                "and rejects out-of-distribution images before classification."
            )
        st.stop()


# ============================================================
# Display: Original image
# ============================================================
st.markdown("---")
st.header("📷 Uploaded Image")
col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    st.image(img_pil, caption=uploaded_file.name, use_container_width=True)


# ============================================================
# Display: Preprocessing comparison
# ============================================================
st.markdown("---")
st.header("🔬 Preprocessing Variants")
st.markdown("The same image after each preprocessing step:")

cols = st.columns(4)
processed_images = {}
for col, cfg_name in zip(cols, CONFIGS):
    use_m, use_c = ABLATION_CONFIGS[cfg_name]
    processed = preprocess(img_resized.copy(), use_median=use_m, use_clahe=use_c)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    processed_images[cfg_name] = processed
    with col:
        st.markdown(f"**{CFG_LABELS[cfg_name]}**")
        st.image(processed_rgb, use_container_width=True)


# ============================================================
# Run all 16 predictions
# ============================================================
st.markdown("---")
st.header("🤖 Predictions from All 16 Model-Configuration Combinations")

predictions = []
probabilities = {}

for clf_name in CLASSIFIERS:
    for cfg_name in CONFIGS:
        feature = extract_hsv_histogram(processed_images[cfg_name]).reshape(1, -1)
        pipe = MODELS[clf_name][cfg_name]

        pred = int(pipe.predict(feature)[0])
        if hasattr(pipe, 'predict_proba'):
            proba = pipe.predict_proba(feature)[0]
        else:
            proba = [0.5, 0.5]
            try:
                df_val = pipe.decision_function(feature)[0]
                proba_1 = 1 / (1 + np.exp(-df_val))
                proba = [1 - proba_1, proba_1]
            except AttributeError:
                pass

        predictions.append({
            'Classifier': clf_name,
            'Configuration': CFG_LABELS[cfg_name],
            'Prediction': CLASS_NAMES[pred].replace('_', ' ').title(),
            'Confidence': float(proba[pred] * 100),
            'P(Healthy)': float(proba[0] * 100),
            'P(Early Signs)': float(proba[1] * 100),
        })
        probabilities[(clf_name, cfg_name)] = proba


# ============================================================
# Prediction matrix
# ============================================================
st.subheader("Prediction Matrix")
st.markdown("Each cell shows the predicted class:")

pred_grid = pd.DataFrame(predictions).pivot(
    index='Classifier', columns='Configuration', values='Prediction'
)
pred_grid = pred_grid[[CFG_LABELS[c] for c in CONFIGS]]
pred_grid = pred_grid.reindex(CLASSIFIERS)


def style_pred(val):
    if 'Healthy' in str(val):
        return 'background-color: #d4f4dd; color: #1b5e20; font-weight: bold;'
    elif 'Early' in str(val):
        return 'background-color: #ffd4d4; color: #b71c1c; font-weight: bold;'
    return ''


styled = pred_grid.style.map(style_pred)
st.dataframe(styled, use_container_width=True)


# ============================================================
# Detailed results
# ============================================================
st.subheader("Detailed Results (with Confidence)")
detail_df = pd.DataFrame(predictions)
detail_df['Confidence'] = detail_df['Confidence'].apply(lambda x: f'{x:.1f}%')
detail_df['P(Healthy)'] = detail_df['P(Healthy)'].apply(lambda x: f'{x:.1f}%')
detail_df['P(Early Signs)'] = detail_df['P(Early Signs)'].apply(lambda x: f'{x:.1f}%')
styled_detail = detail_df.style.map(style_pred, subset=['Prediction'])
st.dataframe(styled_detail, hide_index=True, use_container_width=True)


# ============================================================
# Consensus analysis
# ============================================================
st.markdown("---")
st.header("🧠 Consensus Analysis")

pred_counts = pd.DataFrame(predictions)['Prediction'].value_counts()
healthy_count = int(pred_counts.get('Healthy', 0))
disease_count = int(pred_counts.get('Early Signs', 0))
total = healthy_count + disease_count

col_a, col_b = st.columns(2)
with col_a:
    st.metric(
        "Votes for Healthy",
        f"{healthy_count} / {total}",
        delta=f"{healthy_count/total*100:.0f}% of models"
    )
with col_b:
    st.metric(
        "Votes for Early Signs",
        f"{disease_count} / {total}",
        delta=f"{disease_count/total*100:.0f}% of models"
    )

agreement = max(healthy_count, disease_count) / total
if agreement == 1.0:
    st.success(
        f"✅ **Unanimous consensus**: all 16 model-configuration combinations agree. "
        f"This is a confident, robust prediction."
    )
elif agreement >= 0.75:
    majority = 'Healthy' if healthy_count > disease_count else 'Early Signs'
    st.info(
        f"📊 **Strong majority ({agreement*100:.0f}%)** predicts **{majority}**. "
        f"The remaining models disagree, which suggests this is a moderately "
        f"ambiguous sample. The best-performing combination "
        f"({BEST_CLF} + {CFG_LABELS[BEST_CFG]}) is the most reliable indicator."
    )
else:
    st.warning(
        f"⚠️ **Models disagree substantially.** This is a visually ambiguous "
        f"or borderline image. Trust the best-performing combination "
        f"({BEST_CLF} + {CFG_LABELS[BEST_CFG]}, {BEST_F1:.1f}% F1 on test)."
    )


# ============================================================
# Recommended final diagnosis
# ============================================================
st.markdown("---")
st.header("🎯 Recommended Final Diagnosis")
st.markdown(
    f"Based on the best-performing combination from the benchmark "
    f"(**{BEST_CLF}** + **{CFG_LABELS[BEST_CFG]}**, "
    f"F1 = **{BEST_F1:.1f}%**):"
)

best_proba = probabilities[(BEST_CLF, BEST_CFG)]
best_pred_idx = int(np.argmax(best_proba))
best_pred = CLASS_NAMES[best_pred_idx].replace('_', ' ').title()
best_conf = best_proba[best_pred_idx] * 100

if 'Healthy' in best_pred:
    st.success(f"### Prediction: **{best_pred}** — confidence {best_conf:.1f}%")
else:
    st.error(f"### Prediction: **{best_pred}** — confidence {best_conf:.1f}%")


# ============================================================
# Confidence comparison chart
# ============================================================
st.markdown("---")
st.header("📊 Per-Classifier Confidence Across Configurations")
st.markdown(
    "How does each classifier's confidence in its predicted class change "
    "with different preprocessing? Higher bars = more confident."
)

chart_rows = []
for clf_name in CLASSIFIERS:
    for cfg_name in CONFIGS:
        proba = probabilities[(clf_name, cfg_name)]
        pred_idx = int(np.argmax(proba))
        chart_rows.append({
            'Classifier': clf_name,
            'Configuration': CFG_LABELS[cfg_name],
            'Confidence': proba[pred_idx] * 100,
        })
chart_df = pd.DataFrame(chart_rows)

fig, ax = plt.subplots(figsize=(11, 5))
config_order = [CFG_LABELS[c] for c in CONFIGS]
classifier_order = CLASSIFIERS
chart_matrix = chart_df.pivot(index='Classifier', columns='Configuration', values='Confidence')
chart_matrix = chart_matrix.loc[classifier_order, config_order]

x = np.arange(len(classifier_order))
width = 0.20
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

for i, cfg in enumerate(config_order):
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, chart_matrix[cfg].values, width,
                  label=cfg, color=colors[i], edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, chart_matrix[cfg].values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                f'{val:.0f}', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Classifier', fontweight='bold')
ax.set_ylabel('Confidence (%)', fontweight='bold')
ax.set_title('Predicted-Class Confidence per (Classifier, Configuration)',
             fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classifier_order)
ax.set_ylim(0, 110)
ax.legend(title='Configuration', fontsize=9, loc='lower right')
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
st.pyplot(fig)


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(
    f"Built with Streamlit, OpenCV, scikit-learn, matplotlib, and seaborn. "
    f"Features: 96-dim HSV color histogram. "
    f"Classifiers: SVM, Random Forest, KNN, Logistic Regression. "
    f"Preprocessing variants: baseline, median filter, CLAHE, median + CLAHE. "
    f"OOD detection: One-Class SVM."
)
