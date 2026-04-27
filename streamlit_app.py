"""
Celebrity Face Recognizer — Streamlit App (TFLite Version)
Uses ArcFace TFLite + locally saved gallery files (no internet needed)

Requirements:
    pip install streamlit tensorflow opencv-python pillow scikit-learn numpy

Files needed in SAME folder as this script:
    arcface.tflite
    gallery.pkl (optional for this specific script, but good to keep)
    gallery_matrix.npy
    celeb_names.json

Run:
    streamlit run streamlit_app.py
"""

import os, json, time, io
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# ── CONFIG ─────────────────────────────────────────────────────────────────
MODELS_DIR    = os.path.dirname(os.path.abspath(__file__))
MATRIX_NPY    = os.path.join(MODELS_DIR, "gallery_matrix.npy")
LABELS_JSON   = os.path.join(MODELS_DIR, "celeb_names.json")
MODEL_TFLITE  = os.path.join(MODELS_DIR, "arcface.tflite")

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

@st.cache_resource(show_spinner=False)
def load_all():
    # Load Gallery Data safely
    matrix = np.load(MATRIX_NPY)
    if len(matrix.shape) == 1:
        matrix = np.expand_dims(matrix, axis=0) # ensure 2D array
        
    with open(LABELS_JSON) as f:
        names_raw = json.load(f)
        # Handle if JSON was saved as dict or list
        names = list(names_raw.values()) if isinstance(names_raw, dict) else names_raw
        
    # Initialize TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE)
    interpreter.allocate_tensors()
    
    return matrix, names, interpreter

def run_prediction(img_bytes, matrix, names, interpreter, top_k=5):
    # 1. Load image and ensure RGB
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # 2. Dynamically resize image to what the model strictly expects 
    # Fallback to standard ArcFace dimensions
    req_h, req_w = 112, 112
    try:
        input_details = interpreter.get_input_details()[0]
        input_shape = input_details['shape']
        if len(input_shape) >= 3:
            req_h, req_w = input_shape[1], input_shape[2]
    except Exception:
        pass # use fallback
        
    img = img.resize((req_w, req_h))
    
    # 3. Format array
    img_array = np.array(img, dtype=np.float32)
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)
        
    img_array /= 255.0 

    # 4. Inference
    try:
        # Use signature runner to bypass get_input_details() bug in TF 2.16+
        runner = interpreter.get_signature_runner()
        input_name = list(runner._inputs.keys())[0] if hasattr(runner._inputs, 'keys') else list(runner._inputs)[0][0]
        out_dict = runner(**{input_name: img_array})
        output_name = list(out_dict.keys())[0]
        output_data = out_dict[output_name]
    except Exception:
        # Fallback if signature runner is unavailable
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        interpreter.set_tensor(input_details['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])
    
    # 5. Extract embedding safely (flatten avoids tuple/0D indexing errors)
    vec = output_data.flatten()

    # 7. L2 Normalize
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0: vec_norm = 1e-10
    vec = vec / vec_norm

    # 8. Compute Cosine Similarity
    sims = cosine_similarity(vec.reshape(1, -1), matrix)[0]
    
    # 9. Find Top K Matches safely
    top_k = min(top_k, len(names))
    idx  = np.argsort(sims)[::-1][:top_k]
    
    return [(names[i], float(sims[i])) for i in idx]

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
    <div class="sub">ArcFace TFLite · Deep Metric Learning · 100% Local Inference</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── LOAD MODEL ───────────────────────────────────────────────────────────────
missing = [f for f in [MODEL_TFLITE, MATRIX_NPY, LABELS_JSON] if not os.path.isfile(f)]
if missing:
    st.error(f"Missing model files: {', '.join(os.path.basename(m) for m in missing)}\n\nPlace them in: `{MODELS_DIR}`")
    st.stop()

with st.spinner("Loading ArcFace TFLite model and gallery…"):
    matrix, names, interpreter = load_all()

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
        st.image(img, width='stretch')
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
                                    width='stretch')
    with c2:
        reset_clicked = st.button("↺  Reset", width='stretch')

    st.markdown(f"""
    <div style="margin-top:14px;">
      <span class="chip">◈ TFLite ArcFace</span>
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
        with st.spinner("Running fast TFLite inference…"):
            try:
                t0 = time.time()
                preds = run_prediction(st.session_state.img_bytes, matrix, names, interpreter)
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
        st.markdown('<div class="section-label" style="margin-top:4px;">Top Candidates</div>',
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
  Celebrity Recognizer · TFLite ArcFace Engine · Zero-Dependency Local Inference
</div>""", unsafe_allow_html=True)