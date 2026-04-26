"""
Celebrity Face Recognizer — Tkinter Desktop App
Uses ArcFace + locally saved gallery files (no internet needed)

Requirements (pip install):
    deepface tf-keras opencv-python pillow scikit-learn numpy

Files needed in SAME folder as this script (or update MODELS_DIR below):
    arcface_weights.h5
    gallery.pkl
    gallery_matrix.npy
    celeb_names.json
"""

import os, sys, json, pickle, shutil, threading, time
import numpy as np
import tkinter as tk
from tkinter import filedialog, font as tkfont
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageEnhance
from sklearn.metrics.pairwise import cosine_similarity

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))   # same folder as script
# Override if your files are elsewhere:
# MODELS_DIR = r"C:\Users\YourName\Downloads\ArcFace_FaceRecog"

GALLERY_PKL    = os.path.join(MODELS_DIR, "gallery.pkl")
MATRIX_NPY     = os.path.join(MODELS_DIR, "gallery_matrix.npy")
LABELS_JSON    = os.path.join(MODELS_DIR, "celeb_names.json")
WEIGHTS_SRC    = os.path.join(MODELS_DIR, "arcface_weights.h5")
WEIGHTS_CACHE  = os.path.join(os.path.expanduser("~"), ".deepface", "weights", "arcface_weights.h5")

# ── PALETTE ───────────────────────────────────────────────────────────────────
BG_DEEP    = "#0a0b0f"
BG_PANEL   = "#111318"
BG_CARD    = "#181b23"
BG_HOVER   = "#1e2230"
ACCENT     = "#6ee7f7"       # electric cyan
ACCENT2    = "#a78bfa"       # soft violet
SUCCESS    = "#4ade80"
WARNING    = "#fbbf24"
DANGER     = "#f87171"
TEXT_PRI   = "#f0f4ff"
TEXT_SEC   = "#8892a8"
TEXT_DIM   = "#4a5268"
BORDER     = "#232840"

# ─────────────────────────────────────────────────────────────────────────────
def copy_weights_to_cache():
    """Copy local weights file so DeepFace won't re-download."""
    if os.path.isfile(WEIGHTS_SRC) and not os.path.isfile(WEIGHTS_CACHE):
        os.makedirs(os.path.dirname(WEIGHTS_CACHE), exist_ok=True)
        shutil.copy2(WEIGHTS_SRC, WEIGHTS_CACHE)

def load_gallery():
    with open(GALLERY_PKL, "rb") as f:
        gallery = pickle.load(f)
    matrix = np.load(MATRIX_NPY)
    with open(LABELS_JSON) as f:
        names = json.load(f)
    return gallery, matrix, names

def get_embedding(img_path, model_name="ArcFace"):
    from deepface import DeepFace
    result = DeepFace.represent(
        img_path=img_path,
        model_name=model_name,
        detector_backend="skip",
        enforce_detection=False,
        align=False,
    )
    vec = np.array(result[0]["embedding"], dtype=np.float32)
    return vec / (np.linalg.norm(vec) + 1e-10)

def predict(emb, matrix, names, top_k=5):
    sims = cosine_similarity(emb.reshape(1, -1), matrix)[0]
    idx  = np.argsort(sims)[::-1][:top_k]
    return [(names[i], float(sims[i])) for i in idx]

# ── ROUNDED RECTANGLE HELPER ─────────────────────────────────────────────────
def round_rect(canvas, x1, y1, x2, y2, r=20, **kwargs):
    pts = [x1+r,y1, x2-r,y1, x2,y1, x2,y1+r, x2,y2-r, x2,y2,
           x2-r,y2, x1+r,y2, x1,y2, x1,y2-r, x1,y1+r, x1,y1]
    return canvas.create_polygon(pts, smooth=True, **kwargs)

# ── ANIMATED GRADIENT CANVAS ──────────────────────────────────────────────────
class GlowCanvas(tk.Canvas):
    """Canvas that draws a soft animated glow blob."""
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self._t = 0
        self._animate()

    def _animate(self):
        self.delete("glow")
        w, h = self.winfo_width() or 800, self.winfo_height() or 600
        import math
        cx = w * (0.5 + 0.12 * math.sin(self._t * 0.7))
        cy = h * (0.5 + 0.08 * math.cos(self._t * 0.5))
        for i in range(18, 0, -1):
            r = i * 28
            alpha_hex = format(int(255 * (i / 18) * 0.07), "02x")
            color = f"#1a4a{alpha_hex}" if i % 2 == 0 else f"#2a1a{alpha_hex}"
            # tkinter doesn't support alpha on Canvas ovals — use layered ovals
            # with decreasing brightness instead
            frac = i / 18
            c = self._lerp_color("#0a0b0f", "#1a3a5c", frac * 0.4)
            self.create_oval(cx-r, cy-r, cx+r, cy+r,
                             fill=c, outline="", tags="glow")
        self._t += 0.04
        self.after(40, self._animate)

    @staticmethod
    def _lerp_color(c1, c2, t):
        r1,g1,b1 = int(c1[1:3],16), int(c1[3:5],16), int(c1[5:7],16)
        r2,g2,b2 = int(c2[1:3],16), int(c2[3:5],16), int(c2[5:7],16)
        r = int(r1 + (r2-r1)*t)
        g = int(g1 + (g2-g1)*t)
        b = int(b1 + (b2-b1)*t)
        return f"#{r:02x}{g:02x}{b:02x}"


# ── MAIN APP ──────────────────────────────────────────────────────────────────
class CelebApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Celebrity Face Recognizer  ·  ArcFace")
        self.geometry("980x700")
        self.minsize(860, 620)
        self.configure(bg=BG_DEEP)
        self.resizable(True, True)

        # State
        self._img_path   = None
        self._photo      = None
        self._gallery    = None
        self._matrix     = None
        self._names      = None
        self._loading    = False
        self._model_ready = False

        self._build_ui()
        self._start_model_load()

    # ── BUILD UI ──────────────────────────────────────────────
    def _build_ui(self):
        # Fonts
        try:
            self.fn_title  = tkfont.Font(family="Segoe UI", size=22, weight="bold")
            self.fn_sub    = tkfont.Font(family="Segoe UI", size=10)
            self.fn_label  = tkfont.Font(family="Segoe UI", size=11, weight="bold")
            self.fn_body   = tkfont.Font(family="Segoe UI", size=10)
            self.fn_score  = tkfont.Font(family="Segoe UI", size=13, weight="bold")
            self.fn_rank   = tkfont.Font(family="Segoe UI", size=9)
            self.fn_mono   = tkfont.Font(family="Consolas", size=9)
        except:
            self.fn_title  = tkfont.Font(size=20, weight="bold")
            self.fn_sub    = tkfont.Font(size=9)
            self.fn_label  = tkfont.Font(size=10, weight="bold")
            self.fn_body   = tkfont.Font(size=10)
            self.fn_score  = tkfont.Font(size=12, weight="bold")
            self.fn_rank   = tkfont.Font(size=8)
            self.fn_mono   = tkfont.Font(size=9)

        # ── Header ──────────────────────────────────────────
        header = tk.Frame(self, bg=BG_PANEL, height=64)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        tk.Label(header, text="◈", bg=BG_PANEL, fg=ACCENT,
                 font=tkfont.Font(size=22)).pack(side="left", padx=(20,6), pady=10)
        tk.Label(header, text="Celebrity Recognizer", bg=BG_PANEL,
                 fg=TEXT_PRI, font=self.fn_title).pack(side="left", pady=10)
        tk.Label(header, text="  powered by ArcFace", bg=BG_PANEL,
                 fg=TEXT_DIM, font=self.fn_sub).pack(side="left", pady=18)

        self._status_dot = tk.Label(header, text="●", bg=BG_PANEL,
                                    fg=WARNING, font=tkfont.Font(size=14))
        self._status_dot.pack(side="right", padx=(0,10))
        self._status_lbl = tk.Label(header, text="Loading model…", bg=BG_PANEL,
                                    fg=TEXT_SEC, font=self.fn_body)
        self._status_lbl.pack(side="right")

        sep = tk.Frame(self, bg=BORDER, height=1)
        sep.pack(fill="x")

        # ── Main body ────────────────────────────────────────
        body = tk.Frame(self, bg=BG_DEEP)
        body.pack(fill="both", expand=True, padx=20, pady=16)

        # Left column — image panel
        left = tk.Frame(body, bg=BG_DEEP)
        left.pack(side="left", fill="both", expand=True)

        tk.Label(left, text="INPUT IMAGE", bg=BG_DEEP, fg=TEXT_DIM,
                 font=self.fn_rank).pack(anchor="w")

        # Image drop zone
        self._img_frame = tk.Frame(left, bg=BG_CARD, bd=0,
                                   highlightthickness=2,
                                   highlightbackground=BORDER,
                                   highlightcolor=ACCENT)
        self._img_frame.pack(fill="both", expand=True, pady=(4, 12))

        self._img_canvas = tk.Canvas(self._img_frame, bg=BG_CARD,
                                     bd=0, highlightthickness=0)
        self._img_canvas.pack(fill="both", expand=True)
        self._draw_placeholder()

        # Buttons row
        btn_row = tk.Frame(left, bg=BG_DEEP)
        btn_row.pack(fill="x")

        self._btn_load = self._make_btn(btn_row, "⊕  Load Image",
                                        ACCENT, self._load_image)
        self._btn_load.pack(side="left", expand=True, fill="x", padx=(0,6))

        self._btn_predict = self._make_btn(btn_row, "⚡  Predict",
                                            ACCENT2, self._predict,
                                            disabled=True)
        self._btn_predict.pack(side="left", expand=True, fill="x", padx=(0,6))

        self._btn_reset = self._make_btn(btn_row, "↺  Reset",
                                          TEXT_DIM, self._reset,
                                          flat=True)
        self._btn_reset.pack(side="left", expand=True, fill="x")

        # Right column — results panel
        right = tk.Frame(body, bg=BG_DEEP, width=320)
        right.pack(side="right", fill="y", padx=(16, 0))
        right.pack_propagate(False)

        tk.Label(right, text="RECOGNITION RESULTS", bg=BG_DEEP,
                 fg=TEXT_DIM, font=self.fn_rank).pack(anchor="w")

        self._result_frame = tk.Frame(right, bg=BG_CARD, bd=0,
                                      highlightthickness=1,
                                      highlightbackground=BORDER)
        self._result_frame.pack(fill="both", expand=True, pady=(4, 0))
        self._draw_result_placeholder()

        # Path label at bottom
        self._path_lbl = tk.Label(self, text="No image loaded",
                                  bg=BG_DEEP, fg=TEXT_DIM,
                                  font=self.fn_mono, anchor="w")
        self._path_lbl.pack(fill="x", padx=20, pady=(0,8))

    # ── BUTTON FACTORY ────────────────────────────────────────
    def _make_btn(self, parent, text, color, cmd, disabled=False, flat=False):
        bg   = BG_CARD if flat else BG_HOVER
        fg   = color
        f    = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        btn  = tk.Label(parent, text=text, bg=bg, fg=fg, font=f,
                        cursor="hand2", padx=12, pady=9,
                        relief="flat", bd=0)
        if not disabled:
            btn.bind("<Button-1>", lambda e: cmd())
            btn.bind("<Enter>",    lambda e: btn.config(bg=BG_HOVER if flat else BG_CARD))
            btn.bind("<Leave>",    lambda e: btn.config(bg=BG_CARD if flat else BG_HOVER))
        else:
            btn.config(fg=TEXT_DIM, cursor="")
        btn._disabled = disabled
        btn._cmd = cmd
        return btn

    def _enable_btn(self, btn, color):
        btn.config(fg=color, cursor="hand2")
        btn._disabled = False
        btn.bind("<Button-1>", lambda e: btn._cmd())
        btn.bind("<Enter>",    lambda e: btn.config(bg=BG_HOVER))
        btn.bind("<Leave>",    lambda e: btn.config(bg=BG_CARD))

    # ── PLACEHOLDER DRAWINGS ──────────────────────────────────
    def _draw_placeholder(self):
        c = self._img_canvas
        c.delete("all")
        w, h = max(c.winfo_width(),400), max(c.winfo_height(),300)
        # dashed border
        c.create_rectangle(20, 20, w-20, h-20,
                           outline=BORDER, dash=(8,6), width=2)
        c.create_text(w//2, h//2 - 24, text="⊕",
                      fill=TEXT_DIM, font=tkfont.Font(size=36))
        c.create_text(w//2, h//2 + 18,
                      text="Click  'Load Image'  to begin",
                      fill=TEXT_DIM,
                      font=tkfont.Font(family="Segoe UI", size=11))
        c.create_text(w//2, h//2 + 40,
                      text="Supports JPG · PNG · BMP · WEBP",
                      fill=TEXT_DIM,
                      font=tkfont.Font(family="Segoe UI", size=9))

    def _draw_result_placeholder(self):
        for w in self._result_frame.winfo_children():
            w.destroy()
        lbl = tk.Label(self._result_frame,
                       text="Results will appear here\nafter prediction",
                       bg=BG_CARD, fg=TEXT_DIM,
                       font=self.fn_body, justify="center")
        lbl.place(relx=0.5, rely=0.5, anchor="center")

    # ── MODEL LOADING (background thread) ────────────────────
    def _start_model_load(self):
        threading.Thread(target=self._load_model_thread, daemon=True).start()

    def _load_model_thread(self):
        try:
            self._set_status("Copying weights…", WARNING)
            copy_weights_to_cache()
            self._set_status("Loading gallery…", WARNING)
            g, m, n = load_gallery()
            self._gallery, self._matrix, self._names = g, m, n
            # Warm up DeepFace by importing it
            self._set_status("Warming up ArcFace…", WARNING)
            import deepface  # noqa
            self._model_ready = True
            self._set_status(f"Ready  ·  {len(n)} celebrities", SUCCESS)
            if self._img_path:
                self.after(0, lambda: self._enable_btn(self._btn_predict, ACCENT2))
        except Exception as ex:
            self._set_status(f"Error: {ex}", DANGER)

    def _set_status(self, msg, color):
        self.after(0, lambda: self._status_lbl.config(text=msg))
        self.after(0, lambda: self._status_dot.config(fg=color))

    # ── LOAD IMAGE ────────────────────────────────────────────
    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Select a face image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                       ("All files", "*.*")]
        )
        if not path:
            return
        self._img_path = path
        self._path_lbl.config(text=f"  {os.path.basename(path)}  —  {path}")
        self._show_image(path)
        self._draw_result_placeholder()
        if self._model_ready:
            self._enable_btn(self._btn_predict, ACCENT2)
        else:
            self._set_status("Model still loading… wait a moment", WARNING)

    def _show_image(self, path):
        c = self._img_canvas
        c.update_idletasks()
        w, h = c.winfo_width(), c.winfo_height()
        if w < 10: w, h = 400, 340

        img = Image.open(path).convert("RGB")
        img.thumbnail((w - 20, h - 20), Image.LANCZOS)

        # Subtle glow border
        border_size = 3
        new_w = img.width + border_size*2
        new_h = img.height + border_size*2
        bordered = Image.new("RGB", (new_w, new_h), "#1a3a5c")
        bordered.paste(img, (border_size, border_size))

        self._photo = ImageTk.PhotoImage(bordered)
        c.delete("all")
        c.create_image(w//2, h//2, image=self._photo, anchor="center")

    # ── PREDICT ───────────────────────────────────────────────
    def _predict(self):
        if not self._img_path or not self._model_ready or self._loading:
            return
        self._loading = True
        self._btn_predict.config(text="⏳  Predicting…", fg=TEXT_SEC, cursor="")
        self._draw_result_placeholder()
        threading.Thread(target=self._predict_thread, daemon=True).start()

    def _predict_thread(self):
        try:
            t0 = time.time()
            emb = get_embedding(self._img_path)
            preds = predict(emb, self._matrix, self._names, top_k=5)
            elapsed = time.time() - t0
            self.after(0, lambda: self._show_results(preds, elapsed))
        except Exception as ex:
            self.after(0, lambda: self._show_error(str(ex)))
        finally:
            self._loading = False
            self.after(0, lambda: self._btn_predict.config(
                text="⚡  Predict", fg=ACCENT2, cursor="hand2"))

    def _show_results(self, preds, elapsed):
        for w in self._result_frame.winfo_children():
            w.destroy()

        top_name, top_sim = preds[0]
        conf_pct = top_sim * 100

        # Confidence color
        if conf_pct >= 60:   conf_color = SUCCESS
        elif conf_pct >= 35: conf_color = WARNING
        else:                conf_color = DANGER

        # ── Top result card ──────────────────────────────────
        top_card = tk.Frame(self._result_frame, bg=BG_HOVER, pady=14)
        top_card.pack(fill="x", padx=12, pady=(16, 8))

        tk.Label(top_card, text="BEST MATCH", bg=BG_HOVER,
                 fg=TEXT_DIM, font=self.fn_rank).pack()

        tk.Label(top_card, text=top_name, bg=BG_HOVER,
                 fg=TEXT_PRI, font=tkfont.Font(family="Segoe UI", size=16,
                                               weight="bold"),
                 wraplength=260).pack(pady=(6,2))

        # Confidence bar
        bar_frame = tk.Frame(top_card, bg=BG_HOVER)
        bar_frame.pack(fill="x", padx=20, pady=4)

        bar_bg = tk.Frame(bar_frame, bg=BORDER, height=8)
        bar_bg.pack(fill="x")
        bar_bg.update_idletasks()
        bar_w = max(int(bar_bg.winfo_width() * top_sim), 6)
        bar_fill = tk.Frame(bar_bg, bg=conf_color, height=8, width=bar_w)
        bar_fill.place(x=0, y=0)

        tk.Label(top_card, text=f"{conf_pct:.1f}% confidence",
                 bg=BG_HOVER, fg=conf_color,
                 font=self.fn_score).pack(pady=(4,0))

        # Separator
        tk.Frame(self._result_frame, bg=BORDER, height=1).pack(
            fill="x", padx=12, pady=4)

        # ── Top-5 list ───────────────────────────────────────
        tk.Label(self._result_frame, text="TOP 5 CANDIDATES",
                 bg=BG_CARD, fg=TEXT_DIM, font=self.fn_rank).pack(
                     anchor="w", padx=16, pady=(4,2))

        colors = [ACCENT, ACCENT2, TEXT_PRI, TEXT_SEC, TEXT_DIM]
        medals = ["①", "②", "③", "④", "⑤"]

        for i, (name, sim) in enumerate(preds):
            row = tk.Frame(self._result_frame, bg=BG_CARD, pady=5)
            row.pack(fill="x", padx=12, pady=1)

            tk.Label(row, text=medals[i], bg=BG_CARD,
                     fg=colors[i], font=self.fn_label,
                     width=3).pack(side="left")

            tk.Label(row, text=name, bg=BG_CARD,
                     fg=TEXT_PRI if i == 0 else TEXT_SEC,
                     font=self.fn_label if i == 0 else self.fn_body,
                     anchor="w").pack(side="left", fill="x", expand=True)

            pct = sim * 100
            c2  = SUCCESS if pct >= 60 else (WARNING if pct >= 35 else DANGER)
            tk.Label(row, text=f"{pct:.1f}%", bg=BG_CARD,
                     fg=c2, font=self.fn_body, width=6).pack(side="right")

            # mini bar
            mini_bg = tk.Frame(row, bg=BORDER, height=3, width=80)
            mini_bg.pack(side="right", padx=(4,8))
            mini_bg.update_idletasks()
            fill_w = max(int(80 * sim), 3)
            tk.Frame(mini_bg, bg=c2, height=3, width=fill_w).place(x=0, y=0)

        # Footer
        tk.Frame(self._result_frame, bg=BORDER, height=1).pack(
            fill="x", padx=12, pady=(8,0))
        tk.Label(self._result_frame,
                 text=f"⏱  Inference: {elapsed:.2f}s  ·  Model: ArcFace",
                 bg=BG_CARD, fg=TEXT_DIM, font=self.fn_rank).pack(
                     pady=8)

    def _show_error(self, msg):
        for w in self._result_frame.winfo_children():
            w.destroy()
        tk.Label(self._result_frame,
                 text=f"❌ Error\n\n{msg}",
                 bg=BG_CARD, fg=DANGER,
                 font=self.fn_body, wraplength=280, justify="center").place(
                     relx=0.5, rely=0.5, anchor="center")

    # ── RESET ─────────────────────────────────────────────────
    def _reset(self):
        self._img_path = None
        self._photo    = None
        self._path_lbl.config(text="No image loaded")
        self._img_canvas.delete("all")
        self._draw_placeholder()
        self._draw_result_placeholder()
        self._btn_predict.config(fg=TEXT_DIM, cursor="")
        self._btn_predict._disabled = True


# ── ENTRY POINT ───────────────────────────────────────────────
if __name__ == "__main__":
    # Check files exist
    missing = [f for f in [GALLERY_PKL, MATRIX_NPY, LABELS_JSON]
               if not os.path.isfile(f)]
    if missing:
        import tkinter.messagebox as mb
        root = tk.Tk(); root.withdraw()
        mb.showerror("Missing Files",
            f"Could not find these model files:\n\n" +
            "\n".join(missing) +
            f"\n\nPlace them in:\n{MODELS_DIR}\n\n"
             "Or edit MODELS_DIR at the top of the script.")
        sys.exit(1)

    app = CelebApp()
    app.mainloop()