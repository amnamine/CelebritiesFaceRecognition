"""
Celebrity Face Recognizer — Streamlit App
Uses ArcFace + locally saved gallery files (no internet needed)

Requirements:
    pip install streamlit deepface tf-keras opencv-python pillow scikit-learn numpy

Files needed in SAME folder as this script (or update MODELS_DIR below):
    arcface_weights.h5
    gallery.pkl
    gallery_matrix.npy
    celeb_names.json

Run:
    streamlit run streamlit_app.py
"""

import os, sys, json, pickle, shutil, time, tempfile, io
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# ── CONFIG ─────────────────────────────────────────────────────────────────
MODELS_DIR    = os.path.dirname(os.path.abspath(__file__))
GALLERY_PKL   = os.path.join(MODELS_DIR, "gallery.pkl")
MATRIX_NPY    = os.path.join(MODELS_DIR, "gallery_matrix.npy")
LABELS_JSON   = os.path.join(MODELS_DIR, "celeb_names.json")
WEIGHTS_SRC   = os.path.join(MODELS_DIR, "arcface_weights.h5")
WEIGHTS_CACHE = os.path.join(os.path.expanduser("~"), ".deepface", "weights", "arcface_weights.h5")

# ── PAGE ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Celebrity Recognizer · ArcFace",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── STYLES ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #080a0f !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stAppViewContainer"] > .main { background: #080a0f; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"]    { display: none; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stSidebar"]    { display: none; }
h1,h2,h3,h4,.stMarkdown p   { color: #e8eeff; }

.block-container { padding: 0 !important; max-width: 100% !important; }

.app-header {
    background: #0e1018;
    border-bottom: 1px solid #1e2438;
    padding: 18px 36px;
    display: flex; align-items: center; gap: 14px;
}
.app-header .logo  { font-size: 28px; color: #6ee7f7; }
.app-header .title {
    font-family: 'Syne', sans-serif; font-size: 22px;
    font-weight: 800; color: #f0f4ff; letter-spacing: -0.3px;
}
.app-header .sub { font-size: 11px; color: #4a5268; margin-top: 3px; }

.section-label {
    font-size: 10px; font-weight: 500; letter-spacing: 2px;
    color: #4a5268; text-transform: uppercase; margin-bottom: 8px;
}

[data-testid="stFileUploader"] {
    background: #111520 !important;
    border: 2px dashed #1e2438 !important;
    border-radius: 16px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #6ee7f7 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] p {
    color: #4a5268 !important; font-size: 13px !important;
}

.stButton > button {
    width: 100%; padding: 13px 20px; border-radius: 12px;
    font-family: 'DM Sans', sans-serif; font-weight: 600;
    font-size: 14px; border: none; cursor: pointer;
    transition: all 0.2s; letter-spacing: 0.3px;
}

/* column 1 = Predict */
div[data-testid="column"]:nth-child(1) .stButton > button {
    background: linear-gradient(135deg,#6ee7f7 0%,#a78bfa 100%);
    color: #080a0f; font-weight: 700;
}
div[data-testid="column"]:nth-child(1) .stButton > button:hover {
    box-shadow: 0 8px 28px rgba(110,231,247,0.28);
    transform: translateY(-1px);
}
div[data-testid="column"]:nth-child(1) .stButton > button:disabled {
    background: #1a1e2e !important; color: #3a4060 !important;
}

/* column 2 = Reset */
div[data-testid="column"]:nth-child(2) .stButton > button {
    background: #1a1e2e; color: #8892a8;
    border: 1px solid #1e2438;
}
div[data-testid="column"]:nth-child(2) .stButton > button:hover {
    background: #1e2438; color: #e8eeff;
}

.top-match-card {
    background: linear-gradient(135deg,#111a2e 0%,#131520 100%);
    border: 1px solid #2a3a5c; border-radius: 20px;
    padding: 28px 24px; text-align: center; margin-bottom: 14px;
}
.match-label {
    font-size: 10px; letter-spacing: 3px; color: #4a5268;
    text-transform: uppercase; margin-bottom: 8px;
}
.match-name {
    font-family: 'Syne', sans-serif; font-size: 26px;
    font-weight: 800; color: #f0f4ff; margin: 6px 0; line-height: 1.1;
}
.match-conf { font-size: 38px; font-weight: 700; margin: 6px 0 2px; }
.conf-label { font-size: 10px; color: #4a5268; letter-spacing: 1.5px; }

.bar-wrap {
    background: #1e2438; border-radius: 999px;
    height: 8px; width: 100%; margin: 14px 0 8px; overflow: hidden;
}
.bar-fill { height: 8px; border-radius: 999px; }

.rank-row {
    display: flex; align-items: center; padding: 10px 14px;
    border-radius: 12px; margin-bottom: 6px;
    background: #0e1018; border: 1px solid #1a1e2e; gap: 10px;
}
.rank-medal { font-size: 16px; width: 24px; }
.rank-name  { flex: 1; font-size: 13px; color: #c0c8e0; font-weight: 500; }
.rank-name.top { color: #f0f4ff; font-weight: 700; }
.rank-pct   { font-size: 12px; font-weight: 700; min-width: 48px; text-align: right; }
.mini-bar-wrap {
    width: 60px; height: 4px; border-radius: 999px;
    background: #1e2438; overflow: hidden;
}
.mini-bar-fill { height: 4px; border-radius: 999px; }

.chip {
    display: inline-block; background: #1a1e2e;
    border: 1px solid #232840; border-radius: 999px;
    padding: 4px 12px; font-size: 11px; color: #6ee7f7; margin: 2px 3px;
}
.badge-ready {
    display: inline-block; border-radius: 999px; padding: 4px 14px;
    font-size: 11px; font-weight: 600;
    background: rgba(74,222,128,0.12); color: #4ade80;
}
.placeholder {
    background: #0e1018; border: 2px dashed #1e2438; border-radius: 16px;
    padding: 70px 20px; text-align: center; color: #2a3050; font-size: 13px;
}
.stat-row {
    padding: 12px 16px; background: #0e1018; border-radius: 12px;
    border: 1px solid #1a1e2e; display: flex;
    justify-content: space-between; align-items: center; margin-top: 14px;
}
[data-testid="stImage"] img {
    border-radius: 14px; border: 2px solid #1e2438;
}
.stSpinner > div { border-top-color: #6ee7f7 !important; }
hr { border-color: #1e2438 !important; }
.app-footer {
    text-align: center; padding: 20px; color: #2a3050;
    font-size: 11px; border-top: 1px solid #1e2438; margin-top: 24px;
}
</style>
""", unsafe_allow_html=True)

# ── HELPERS ─────────────────────────────────────────────────────────────────
def conf_color(pct):
    if pct >= 60:   return "#4ade80"
    elif pct >= 35: return "#fbbf24"
    return "#f87171"

def conf_gradient(pct):
    if pct >= 60:   return "linear-gradient(90deg,#4ade80,#22d3ee)"
    elif pct >= 35: return "linear-gradient(90deg,#fbbf24,#f97316)"
    return "linear-gradient(90deg,#f87171,#e879f9)"

def copy_weights():
    if os.path.isfile(WEIGHTS_SRC) and not os.path.isfile(WEIGHTS_CACHE):
        os.makedirs(os.path.dirname(WEIGHTS_CACHE), exist_ok=True)
        shutil.copy2(WEIGHTS_SRC, WEIGHTS_CACHE)

@st.cache_resource(show_spinner=False)
def load_all():
    copy_weights()
    with open(GALLERY_PKL, "rb") as f:
        gallery = pickle.load(f)
    matrix = np.load(MATRIX_NPY)
    with open(LABELS_JSON) as f:
        names = json.load(f)
    return gallery, matrix, names

def run_prediction(img_bytes, suffix, matrix, names, top_k=5):
    from deepface import DeepFace
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name
    try:
        result = DeepFace.represent(
            img_path=tmp_path,
            model_name="ArcFace",
            detector_backend="skip",
            enforce_detection=False,
            align=False,
        )
        vec = np.array(result[0]["embedding"], dtype=np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        sims = cosine_similarity(vec.reshape(1, -1), matrix)[0]
        idx  = np.argsort(sims)[::-1][:top_k]
        return [(names[i], float(sims[i])) for i in idx]
    finally:
        os.unlink(tmp_path)

# ── SESSION STATE ────────────────────────────────────────────────────────────
for k, v in [("results", None), ("elapsed", None),
             ("img_bytes", None), ("img_name", ""), ("reset_key", 0)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <span class="logo">◈</span>
  <div>
    <div class="title">Celebrity Face Recognizer</div>
    <div class="sub">ArcFace · Deep Metric Learning · 100% Local Inference</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── LOAD MODEL ───────────────────────────────────────────────────────────────
missing = [f for f in [GALLERY_PKL, MATRIX_NPY, LABELS_JSON] if not os.path.isfile(f)]
if missing:
    st.error(f"Missing model files: {', '.join(os.path.basename(m) for m in missing)}\n\nPlace them in: `{MODELS_DIR}`")
    st.stop()

with st.spinner("Loading ArcFace gallery…"):
    gallery, matrix, names = load_all()

# ── LAYOUT ───────────────────────────────────────────────────────────────────
st.markdown("<div style='padding: 28px 36px 0;'>", unsafe_allow_html=True)
col_left, col_right = st.columns([1.1, 1], gap="large")

# ──────────────── LEFT ───────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="section-label">Input Image</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="drop",
        type=["jpg","jpeg","png","bmp","webp"],
        key=f"up_{st.session_state.reset_key}",
        label_visibility="collapsed",
    )

    if uploaded:
        raw = uploaded.read()
        st.session_state.img_bytes = raw
        st.session_state.img_name  = uploaded.name
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        st.image(img, use_container_width=True)
        st.markdown(
            f'<div style="margin-top:8px;">'
            f'<span class="chip">📄 {uploaded.name}</span>'
            f'<span class="chip">📐 {img.width}×{img.height}px</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""
        <div class="placeholder">
          <div style="font-size:44px;margin-bottom:14px;color:#1e2438;">⊕</div>
          <div style="font-size:14px;">Drop a face image here</div>
          <div style="margin-top:6px;font-size:11px;color:#1e2438;">
            JPG · PNG · BMP · WEBP
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="small")
    with c1:
        predict_clicked = st.button("⚡  Predict",
                                    disabled=(not uploaded),
                                    use_container_width=True)
    with c2:
        reset_clicked = st.button("↺  Reset", use_container_width=True)

    st.markdown(f"""
    <div style="margin-top:14px;">
      <span class="chip">◈ ArcFace</span>
      <span class="chip">👤 {len(names)} celebrities</span>
      <span class="badge-ready" style="margin-left:6px;">● Ready</span>
    </div>""", unsafe_allow_html=True)

# ──────────────── RIGHT ──────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="section-label">Recognition Results</div>',
                unsafe_allow_html=True)

    if reset_clicked:
        st.session_state.results   = None
        st.session_state.elapsed   = None
        st.session_state.img_bytes = None
        st.session_state.img_name  = ""
        st.session_state.reset_key += 1
        st.rerun()

    if predict_clicked and st.session_state.img_bytes:
        with st.spinner("Running ArcFace inference…"):
            try:
                t0 = time.time()
                suffix = os.path.splitext(st.session_state.img_name)[-1] or ".jpg"
                preds = run_prediction(st.session_state.img_bytes,
                                       suffix, matrix, names)
                st.session_state.results = preds
                st.session_state.elapsed = time.time() - t0
            except Exception as ex:
                st.error(f"❌ Prediction failed: {ex}")

    if st.session_state.results:
        preds = st.session_state.results
        top_name, top_sim = preds[0]
        top_pct = top_sim * 100
        cc = conf_color(top_pct)
        cg = conf_gradient(top_pct)

        # Top card
        st.markdown(f"""
        <div class="top-match-card">
          <div class="match-label">✦ Best Match</div>
          <div class="match-name">{top_name}</div>
          <div class="bar-wrap">
            <div class="bar-fill" style="width:{int(top_pct)}%;background:{cg};"></div>
          </div>
          <div class="match-conf" style="color:{cc};">{top_pct:.1f}%</div>
          <div class="conf-label">CONFIDENCE SCORE</div>
        </div>""", unsafe_allow_html=True)

        # Top 5
        medals = ["①","②","③","④","⑤"]
        st.markdown('<div class="section-label" style="margin-top:4px;">Top 5 Candidates</div>',
                    unsafe_allow_html=True)

        rows = ""
        for i, (name, sim) in enumerate(preds):
            pct    = sim * 100
            c      = conf_color(pct)
            mini_w = int(pct * 0.6)
            nc     = "rank-name top" if i == 0 else "rank-name"
            rows += f"""
            <div class="rank-row">
              <span class="rank-medal">{medals[i]}</span>
              <span class="{nc}">{name}</span>
              <div class="mini-bar-wrap">
                <div class="mini-bar-fill" style="width:{mini_w}px;background:{c};"></div>
              </div>
              <span class="rank-pct" style="color:{c};">{pct:.1f}%</span>
            </div>"""

        st.markdown(rows, unsafe_allow_html=True)

        # Stats footer
        st.markdown(f"""
        <div class="stat-row">
          <span style="font-size:11px;color:#4a5268;">⏱ Inference time</span>
          <span style="font-size:13px;color:#6ee7f7;font-weight:700;">
            {st.session_state.elapsed:.3f}s
          </span>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="placeholder">
          <div style="font-size:40px;margin-bottom:14px;color:#1e2438;">◈</div>
          <div style="font-size:14px;">Load an image and click Predict</div>
          <div style="margin-top:6px;font-size:11px;color:#1e2438;">
            Results appear here
          </div>
        </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="app-footer">
  Celebrity Recognizer · ArcFace Deep Metric Learning · Local Inference · No Internet Required
</div>""", unsafe_allow_html=True)