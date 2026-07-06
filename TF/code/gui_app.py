"""GUI (Gradio) para el usuario final: sube un video corto y el sistema
responde las 3 preguntas de negocio del caso de uso, usando ambos modelos
(HOG+SVM y ResNet50) sobre una muestra de frames del clip.

Requiere haber corrido antes train_hog_svm.py y train_resnet50.py (para
tener models/hog_svm.pkl y models/resnet50.pt).

Uso:
    python gui_app.py
"""

from collections import Counter
from pathlib import Path

import cv2
import gradio as gr
import joblib
import numpy as np
import torch
from PIL import Image
from skimage.feature import hog
from torch import nn
from torchvision import models, transforms

from config import (
    CLASS_TO_BUSINESS_LABEL,
    CLASSES,
    FRAME_SIZE,
    HOG_CELLS_PER_BLOCK,
    HOG_IMAGE_SIZE,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_FRAMES_PER_VIDEO,
    MODELS_DIR,
    RESNET_IMAGE_SIZE,
)

IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

# --- Carga de modelos ---
hog_bundle = joblib.load(MODELS_DIR / "hog_svm.pkl")
hog_svm = hog_bundle["model"]
hog_scaler = hog_bundle["scaler"]

resnet = models.resnet50()
resnet.fc = nn.Linear(resnet.fc.in_features, len(CLASSES))
resnet.load_state_dict(torch.load(MODELS_DIR / "resnet50.pt", map_location="cpu"))
resnet.eval()

resnet_transform = transforms.Compose([
    transforms.Resize(RESNET_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ============================== Inferencia ==================================

def sample_frames_from_video(video_path: str, max_frames: int = MAX_FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()

    if not frames:
        return []
    if len(frames) <= max_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
    return [frames[i] for i in indices]


def predict_hog_svm(frames_bgr) -> tuple[str, float]:
    height, width = HOG_IMAGE_SIZE
    feats = []
    for frame in frames_bgr:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
        feats.append(hog(
            gray, orientations=HOG_ORIENTATIONS, pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK, block_norm="L2-Hys",
        ))
    X = hog_scaler.transform(np.array(feats))
    preds = hog_svm.predict(X)
    probs = hog_svm.predict_proba(X)

    vote = Counter(preds).most_common(1)[0][0]
    avg_conf = probs[:, list(hog_svm.classes_).index(vote)].mean()
    return vote, float(avg_conf)


@torch.no_grad()
def predict_resnet(frames_bgr) -> tuple[str, float, list[str]]:
    tensors = []
    for frame in frames_bgr:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensors.append(resnet_transform(img))
    batch = torch.stack(tensors)

    logits = resnet(batch)
    probs = torch.softmax(logits, dim=1).numpy()
    preds_idx = probs.argmax(axis=1)
    preds = [IDX_TO_CLASS[i] for i in preds_idx]

    vote = Counter(preds).most_common(1)[0][0]
    vote_idx = CLASSES.index(vote)
    avg_conf = probs[:, vote_idx].mean()
    return vote, float(avg_conf), preds


def business_answers(vote_class: str, frame_preds: list[str]) -> tuple[str, str, str, str]:
    """P1 y P2 se basan en la clase ganadora del clip (mayoria de votos).

    P3 (cuello de botella) sigue la definicion del Hito 1: proporcion de
    frames 'stand' vs 'pick' dentro del propio clip, no solo la clase
    ganadora. Umbrales: >=50% frames 'stand' -> Si; mas 'stand' que 'pick'
    pero sin llegar a mayoria absoluta -> Proximo; en otro caso -> No.
    """
    label = CLASS_TO_BUSINESS_LABEL[vote_class]
    p1 = label
    p2 = {"pick": "Alto", "run": "Medio", "stand": "Bajo"}[vote_class]

    total = len(frame_preds)
    stand_ratio = frame_preds.count("stand") / total
    pick_ratio = frame_preds.count("pick") / total

    if stand_ratio >= 0.5:
        p3 = "Si"
    elif stand_ratio > pick_ratio:
        p3 = "Proximo"
    else:
        p3 = "No"

    detail = f"{stand_ratio:.0%} frames en Standing &nbsp;•&nbsp; {pick_ratio:.0%} frames en Pick Up"
    return p1, p2, p3, detail


# ============================== Presentacion =================================

CLASS_COLOR = {"pick": "#8b5cf6", "run": "#22d3ee", "stand": "#f59e0b"}
P2_BADGE = {"Alto": "good", "Medio": "warn", "Bajo": "bad"}
P3_BADGE = {"No": "good", "Proximo": "warn", "Si": "bad"}

EMPTY_STATE_HTML = """
<div class="empty-state">
  <div class="empty-mark"></div>
  <p>Sube o elige un video de ejemplo para analizarlo.</p>
</div>
"""


def render_error(message: str) -> str:
    return f"""
    <div class="empty-state">
      <div class="empty-mark error-mark"></div>
      <p>{message}</p>
    </div>
    """


def class_dot(class_name: str) -> str:
    return f'<span class="dot" style="background:{CLASS_COLOR[class_name]}"></span>'


def status_dot(style: str) -> str:
    return f'<span class="dot dot-{style}"></span>'


def render_results_html(hog_class, hog_conf, resnet_class, resnet_conf, p1, p2, p3, ratio_detail, n_frames) -> str:
    p2_style = P2_BADGE[p2]
    p3_style = P3_BADGE[p3]

    return f"""
    <div class="results-panel">
      <div class="section-title">Comparacion de modelos</div>
      <div class="model-row">
        <div class="model-name">HOG + SVM</div>
        <div class="conf-bar"><div class="conf-fill hog" style="width:{hog_conf*100:.0f}%"></div></div>
        <div class="conf-label">{class_dot(hog_class)}{hog_class} &nbsp;{hog_conf:.0%}</div>
      </div>
      <div class="model-row">
        <div class="model-name">ResNet50</div>
        <div class="conf-bar"><div class="conf-fill resnet" style="width:{resnet_conf*100:.0f}%"></div></div>
        <div class="conf-label">{class_dot(resnet_class)}{resnet_class} &nbsp;{resnet_conf:.0%}</div>
      </div>

      <div class="section-title" style="margin-top:6px;">Respuestas de negocio (segun ResNet50)</div>
      <div class="answers-grid">
        <div class="answer-card badge-{p2_style}">
          <div class="label">P1 &middot; Estado</div>
          <div class="value">{status_dot(p2_style)}{p1}</div>
        </div>
        <div class="answer-card badge-{p2_style}">
          <div class="label">P2 &middot; Productividad</div>
          <div class="value">{status_dot(p2_style)}{p2}</div>
        </div>
        <div class="answer-card badge-{p3_style}">
          <div class="label">P3 &middot; Cuello de botella</div>
          <div class="value">{status_dot(p3_style)}{p3}</div>
        </div>
      </div>

      <div class="detail-footer">{n_frames} frames analizados &nbsp;&middot;&nbsp; {ratio_detail}</div>
    </div>
    """


def classify_video(video_path: str) -> str:
    if video_path is None:
        return EMPTY_STATE_HTML

    frames = sample_frames_from_video(video_path)
    if not frames:
        return render_error("No se pudieron leer frames de este video.")

    hog_class, hog_conf = predict_hog_svm(frames)
    resnet_class, resnet_conf, resnet_frame_preds = predict_resnet(frames)

    # Para la respuesta final al usuario se prioriza el modelo profundo
    # (mayor precision en la comparativa, ver DECISIONES_TECNICAS.md),
    # mostrando ambos resultados para transparencia.
    p1, p2, p3, ratio_detail = business_answers(resnet_class, resnet_frame_preds)

    return render_results_html(
        hog_class, hog_conf, resnet_class, resnet_conf, p1, p2, p3, ratio_detail, len(frames),
    )


# ============================== Interfaz =====================================

AURORA_HTML = """
<div class="aurora-wrap">
  <div class="aurora-blob b1"></div>
  <div class="aurora-blob b2"></div>
  <div class="aurora-blob b3"></div>
  <div class="aurora-blob b4"></div>
</div>
"""

CUSTOM_CSS = """
:root {
  --glass-bg: rgba(255, 255, 255, 0.06);
  --glass-border: rgba(255, 255, 255, 0.14);
}

body, .gradio-container { background: #05060a !important; }

.aurora-wrap { position: fixed; inset: 0; z-index: 0; overflow: hidden; pointer-events: none; }
.aurora-blob { position: absolute; border-radius: 50%; filter: blur(90px); opacity: 0.5; }
.b1 { width: 480px; height: 480px; background: #7c3aed; top: -120px; left: -100px; animation: drift1 22s ease-in-out infinite; }
.b2 { width: 520px; height: 520px; background: #06b6d4; bottom: -160px; right: -120px; animation: drift2 26s ease-in-out infinite; }
.b3 { width: 380px; height: 380px; background: #db2777; top: 25%; right: 10%; animation: drift3 20s ease-in-out infinite; opacity: 0.4; }
.b4 { width: 340px; height: 340px; background: #22c55e; bottom: 8%; left: 15%; animation: drift1 30s ease-in-out infinite reverse; opacity: 0.25; }

@keyframes drift1 { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(70px,50px) scale(1.15); } }
@keyframes drift2 { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(-60px,-40px) scale(1.1); } }
@keyframes drift3 { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(-40px,60px) scale(0.9); } }

.gradio-container { position: relative; z-index: 1; }

.hero { text-align: center; padding: 22px 0 4px 0; }
.hero h1 {
  font-size: 2.1rem; font-weight: 700; margin: 0;
  background: linear-gradient(90deg, #a78bfa, #22d3ee 55%, #34d399);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { color: rgba(255,255,255,0.55); margin-top: 6px; font-size: 0.95rem; }

.glass-card {
  background: var(--glass-bg) !important;
  backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
  border: 1px solid var(--glass-border) !important;
  border-radius: 20px !important;
  box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}

.results-panel { display: flex; flex-direction: column; gap: 12px; animation: fadeIn 0.5s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

.section-title {
  font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;
  color: rgba(255,255,255,0.45); font-weight: 600;
}

.model-row { display: flex; align-items: center; gap: 10px; }
.model-name { width: 110px; font-size: 0.85rem; color: rgba(255,255,255,0.75); font-weight: 600; }
.conf-bar { flex: 1; height: 10px; background: rgba(255,255,255,0.08); border-radius: 999px; overflow: hidden; }
.conf-fill { height: 100%; border-radius: 999px; transition: width 0.6s ease; }
.conf-fill.hog { background: linear-gradient(90deg,#8b5cf6,#c084fc); }
.conf-fill.resnet { background: linear-gradient(90deg,#22d3ee,#34d399); }
.conf-label { width: 150px; text-align: right; font-size: 0.8rem; color: rgba(255,255,255,0.85); white-space: nowrap; }

.answers-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.answer-card { border-radius: 16px; padding: 16px 8px; text-align: center; border: 1px solid rgba(255,255,255,0.1); }
.answer-card .label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; color: rgba(255,255,255,0.5); }
.answer-card .value {
  display: flex; align-items: center; justify-content: center; gap: 7px;
  font-size: 0.95rem; font-weight: 700; margin-top: 6px; color: rgba(255,255,255,0.92);
}

.badge-good { background: rgba(52,211,153,0.12); border-color: rgba(52,211,153,0.25) !important; }
.badge-warn { background: rgba(251,191,36,0.12); border-color: rgba(251,191,36,0.25) !important; }
.badge-bad { background: rgba(248,113,113,0.14); border-color: rgba(248,113,113,0.3) !important; }

.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.dot-good { background: #34d399; box-shadow: 0 0 8px rgba(52,211,153,0.7); }
.dot-warn { background: #fbbf24; box-shadow: 0 0 8px rgba(251,191,36,0.7); }
.dot-bad { background: #f87171; box-shadow: 0 0 8px rgba(248,113,113,0.7); }

.detail-footer { font-size: 0.75rem; color: rgba(255,255,255,0.45); text-align: center; margin-top: 2px; }

.empty-state { text-align: center; padding: 52px 10px; color: rgba(255,255,255,0.4); }
.empty-mark {
  width: 46px; height: 46px; margin: 0 auto 14px auto;
  border: 2px solid rgba(255,255,255,0.25); border-radius: 50%;
  position: relative;
}
.empty-mark::after {
  content: ""; position: absolute; top: 50%; left: 55%;
  width: 12px; height: 12px; border-right: 2px solid rgba(255,255,255,0.25);
  border-bottom: 2px solid rgba(255,255,255,0.25);
  transform: translate(-50%,-65%) rotate(-45deg);
}
.empty-mark.error-mark { border-color: rgba(248,113,113,0.5); }
.empty-mark.error-mark::after { border-color: transparent; }
.empty-mark.error-mark::before {
  content: "!"; position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  color: #f87171; font-weight: 700; font-size: 1.1rem;
}
"""

theme = gr.themes.Glass(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Sora"), "ui-sans-serif", "system-ui"],
)

sample_videos = [str(p) for p in sorted(SAMPLES_DIR.glob("*.mp4"))] if SAMPLES_DIR.exists() else []

with gr.Blocks(title="Monitor de Productividad en Almacen") as demo:
    gr.HTML(AURORA_HTML)
    gr.HTML("""
    <div class="hero">
      <h1>Monitor de Productividad en Almacen</h1>
      <p>Clasificacion de acciones humanas en video &nbsp;&middot;&nbsp; HOG+SVM vs. ResNet50</p>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, elem_classes="glass-card"):
            video_input = gr.Video(label="Video corto del empleado (clip de almacen)")
            analyze_btn = gr.Button("Analizar video", variant="primary")
            if sample_videos:
                gr.Examples(examples=sample_videos, inputs=video_input, label="Videos de ejemplo (pick / run / stand)")

        with gr.Column(scale=1, elem_classes="glass-card"):
            results_html = gr.HTML(value=EMPTY_STATE_HTML)

    analyze_btn.click(fn=classify_video, inputs=video_input, outputs=results_html)
    video_input.change(fn=classify_video, inputs=video_input, outputs=results_html)


if __name__ == "__main__":
    demo.launch(theme=theme, css=CUSTOM_CSS)
