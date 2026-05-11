import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import time
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chip Defect Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark industrial background */
.stApp {
    background-color: #0d0f14;
    color: #e8e8e8;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #13161d;
    border-right: 1px solid #2a2d36;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #1a1d26 0%, #0d0f14 100%);
    border: 1px solid #2a2d36;
    border-left: 4px solid #f5a623;
    padding: 20px 28px;
    border-radius: 8px;
    margin-bottom: 24px;
}
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 28px;
    color: #f5f5f5;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}
.main-header p {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #888;
    margin: 0;
    letter-spacing: 0.5px;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
}
.metric-card {
    flex: 1;
    padding: 20px;
    border-radius: 8px;
    border: 1px solid #2a2d36;
    background: #13161d;
    text-align: center;
}
.metric-card.whole {
    border-top: 3px solid #22c55e;
}
.metric-card.broken {
    border-top: 3px solid #ef4444;
}
.metric-card.total {
    border-top: 3px solid #f5a623;
}
.metric-card .val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 42px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 6px;
}
.metric-card.whole .val  { color: #22c55e; }
.metric-card.broken .val { color: #ef4444; }
.metric-card.total .val  { color: #f5a623; }
.metric-card .lbl {
    font-size: 11px;
    color: #888;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 16px;
}
.status-badge.ok     { background: #14532d; color: #4ade80; border: 1px solid #22c55e; }
.status-badge.warn   { background: #450a0a; color: #f87171; border: 1px solid #ef4444; }

/* Section label */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #f5a623;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #2a2d36;
}

/* Per-class breakdown */
.class-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px;
    border-radius: 6px;
    background: #1a1d26;
    margin-bottom: 8px;
    border: 1px solid #2a2d36;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}
.class-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
}

/* Model info box */
.info-box {
    background: #13161d;
    border: 1px solid #2a2d36;
    border-radius: 8px;
    padding: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #aaa;
    line-height: 1.8;
}

/* Webcam hint box */
.webcam-hint {
    background: #0f1a2e;
    border: 1px solid #1e3a5f;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 14px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #93c5fd;
    margin-bottom: 16px;
}

/* Streamlit overrides */
.stSlider > div > div { background: #2a2d36; }
div[data-testid="stFileUploader"] {
    background: #13161d;
    border: 1px dashed #2a2d36;
    border-radius: 8px;
}
.stButton > button {
    background: #f5a623;
    color: #000;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    border: none;
    border-radius: 6px;
    padding: 10px 24px;
    font-size: 13px;
    letter-spacing: 0.5px;
    width: 100%;
}
.stButton > button:hover { background: #e09415; color: #000; }

div[data-testid="stImage"] img {
    border-radius: 8px;
    border: 1px solid #2a2d36;
}

/* Camera input styling */
div[data-testid="stCameraInput"] > div {
    border-radius: 8px;
    border: 1px solid #2a2d36 !important;
    background: #13161d !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
WHOLE_COLOR   = (34, 197, 94)    # green  (BGR for OpenCV)
BROKEN_COLOR  = (68, 68, 239)    # red    (BGR for OpenCV)
CLASS_NAMES   = {0: "broken_chip", 1: "whole_chip"}
CLASS_COLORS  = {0: BROKEN_COLOR, 1: WHOLE_COLOR}

# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    return YOLO(path)

# ── Inference on single frame ──────────────────────────────────────────────────
def run_inference(model, frame_bgr, conf_thresh):
    results = model.predict(
        source     = frame_bgr,
        conf       = conf_thresh,
        verbose    = False,
        device     = "cpu"
    )[0]

    whole_count  = 0
    broken_count = 0
    annotated    = frame_bgr.copy()
    h, w         = annotated.shape[:2]
    box_thick    = max(2, int(min(w, h) * 0.003))
    font_scale   = max(0.45, min(w, h) * 0.001)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color     = CLASS_COLORS.get(cls_id, (200, 200, 200))
        cls_name  = CLASS_NAMES.get(cls_id, "unknown")
        label     = f"{cls_name}  {conf:.2f}"

        if cls_id == 1:
            whole_count  += 1
        else:
            broken_count += 1

        # Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thick)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        lbl_y1 = max(0, y1 - th - 8)
        lbl_y2 = y1
        cv2.rectangle(annotated, (x1, lbl_y1), (x1 + tw + 8, lbl_y2), color, -1)

        # Label text
        cv2.putText(
            annotated, label,
            (x1 + 4, lbl_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (255, 255, 255), 2, cv2.LINE_AA
        )

    return annotated, whole_count, broken_count

# ── Shared result renderer ─────────────────────────────────────────────────────
def render_results(bgr_img, annotated, whole_c, broken_c, elapsed_ms):
    """Render detection metrics, images, breakdown, optional audio, and download."""
    total_c = whole_c + broken_c

    # Metrics
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card whole">
            <div class="val">{whole_c}</div>
            <div class="lbl">Whole chips</div>
        </div>
        <div class="metric-card broken">
            <div class="val">{broken_c}</div>
            <div class="lbl">Broken chips</div>
        </div>
        <div class="metric-card total">
            <div class="val">{total_c}</div>
            <div class="lbl">Total detected</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status badge
    if broken_c > 0:
        badge_cls  = "warn"
        badge_text = f"⚠ {broken_c} BROKEN CHIP{'S' if broken_c>1 else ''} DETECTED"
    else:
        badge_cls  = "ok"
        badge_text = "✓ ALL CHIPS INTACT"

    st.markdown(f'<span class="status-badge {badge_cls}">{badge_text}</span>', unsafe_allow_html=True)

    # Side by side images
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-label">Original</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.markdown('<div class="section-label">Detected</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Class breakdown
    st.markdown('<div class="section-label" style="margin-top:16px">Detection breakdown</div>', unsafe_allow_html=True)
    whole_pct  = int((whole_c  / total_c * 100) if total_c > 0 else 0)
    broken_pct = int((broken_c / total_c * 100) if total_c > 0 else 0)

    st.markdown(f"""
    <div class="class-row">
        <span><span class="class-dot" style="background:#22c55e"></span>whole_chip</span>
        <span style="color:#22c55e">{whole_c} detections &nbsp;·&nbsp; {whole_pct}%</span>
    </div>
    <div class="class-row">
        <span><span class="class-dot" style="background:#ef4444"></span>broken_chip</span>
        <span style="color:#ef4444">{broken_c} detections &nbsp;·&nbsp; {broken_pct}%</span>
    </div>
    <div class="class-row" style="margin-top:4px">
        <span style="color:#888">⏱ Inference time</span>
        <span style="color:#f5a623;font-family:'JetBrains Mono',monospace">{elapsed_ms:.1f} ms</span>
    </div>
    """, unsafe_allow_html=True)

    # Download annotated image
    st.markdown('<div class="section-label" style="margin-top:16px">Export</div>', unsafe_allow_html=True)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    pil_img       = Image.fromarray(annotated_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    st.download_button(
        label     = "⬇ Download annotated image",
        data      = buf.getvalue(),
        file_name = "chip_detection_result.png",
        mime      = "image/png"
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">⚙ Model</div>', unsafe_allow_html=True)
    model_path = st.text_input("best.pt path", value="best.pt", label_visibility="collapsed")

    st.markdown('<div class="section-label" style="margin-top:16px">⚙ Confidence threshold</div>', unsafe_allow_html=True)
    conf_thresh = st.slider("Confidence", 0.10, 0.95, 0.35, 0.05, label_visibility="collapsed")

    st.markdown('<div class="section-label" style="margin-top:16px">⚙ Input type</div>', unsafe_allow_html=True)
    input_type = st.radio(
        "Input",
        ["📷 Image", "🎬 Video", "📸 Webcam"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="section-label" style="margin-top:16px">ℹ Model info</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
    
    Classes → broken_chip, whole_chip<br>
    Conf   → {conf_thresh:.2f}<br>
    Device → CPU
    </div>
    """, unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 Potato Chip Defect Detector</h1>
    <p>COMPUTER VISION · REAL-TIME QUALITY INSPECTION</p>
</div>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
if not os.path.exists(model_path):
    st.error(f"❌ Model not found at `{model_path}`. Place `best.pt` in the same folder as `app.py`.")
    st.stop()

model = load_model(model_path)
st.markdown('<span class="status-badge ok">● MODEL LOADED</span>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE MODE
# ══════════════════════════════════════════════════════════════════════════════
if input_type == "📷 Image":
    uploaded = st.file_uploader(
        "Upload a chip image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        bgr_img    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Running inference..."):
            t0 = time.time()
            annotated, whole_c, broken_c = run_inference(model, bgr_img, conf_thresh)
            elapsed = (time.time() - t0) * 1000

        render_results(bgr_img, annotated, whole_c, broken_c, elapsed)

    else:
        st.info("👆 Upload a chip image from the file uploader above to start detection.")


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO MODE
# ══════════════════════════════════════════════════════════════════════════════
elif input_type == "🎬 Video":
    uploaded_video = st.file_uploader(
        "Upload a chip video",
        type=["mp4", "mov", "avi", "mkv"],
        label_visibility="collapsed"
    )

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()
        tfile.close()
        input_path = tfile.name

        cap          = cv2.VideoCapture(input_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        st.markdown(f"""
        <div class="info-box" style="margin-bottom:16px">
        Resolution → {width}×{height} &nbsp;·&nbsp;
        FPS → {fps:.1f} &nbsp;·&nbsp;
        Frames → {total_frames}
        </div>
        """, unsafe_allow_html=True)

        if st.button("▶ Run Detection on Video"):
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

            cap    = cv2.VideoCapture(input_path)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out    = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            preview_placeholder = st.empty()
            progress_bar        = st.progress(0, text="Processing frames...")
            metrics_placeholder = st.empty()

            total_whole  = 0
            total_broken = 0
            frame_idx    = 0
            t_start      = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, w_c, b_c = run_inference(model, frame, conf_thresh)
                total_whole  += w_c
                total_broken += b_c
                frame_idx    += 1

                out.write(annotated)

                if frame_idx % 8 == 0 or frame_idx == 1:
                    rgb_preview = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    preview_placeholder.image(rgb_preview, use_container_width=True, caption=f"Frame {frame_idx}/{total_frames}")

                    pct = frame_idx / total_frames
                    progress_bar.progress(pct, text=f"Processing frame {frame_idx} of {total_frames}...")

                    metrics_placeholder.markdown(f"""
                    <div class="metric-row">
                        <div class="metric-card whole">
                            <div class="val">{total_whole}</div>
                            <div class="lbl">Whole (cumulative)</div>
                        </div>
                        <div class="metric-card broken">
                            <div class="val">{total_broken}</div>
                            <div class="lbl">Broken (cumulative)</div>
                        </div>
                        <div class="metric-card total">
                            <div class="val">{frame_idx}</div>
                            <div class="lbl">Frames processed</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            cap.release()
            out.release()

            with open(out_path, "rb") as f:
                video_bytes = f.read()

            for tmp in [input_path, out_path]:
                try:
                    os.unlink(tmp)
                except PermissionError:
                    pass

            elapsed_total = time.time() - t_start
            progress_bar.empty()

            total_detections = total_whole + total_broken
            st.markdown("---")
            st.markdown('<div class="section-label">Final Summary</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card whole">
                    <div class="val">{total_whole}</div>
                    <div class="lbl">Whole chips</div>
                </div>
                <div class="metric-card broken">
                    <div class="val">{total_broken}</div>
                    <div class="lbl">Broken chips</div>
                </div>
                <div class="metric-card total">
                    <div class="val">{total_detections}</div>
                    <div class="lbl">Total detections</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            whole_pct  = int((total_whole  / total_detections * 100) if total_detections > 0 else 0)
            broken_pct = int((total_broken / total_detections * 100) if total_detections > 0 else 0)
            avg_fps    = total_frames / elapsed_total

            st.markdown(f"""
            <div class="class-row">
                <span><span class="class-dot" style="background:#22c55e"></span>whole_chip</span>
                <span style="color:#22c55e">{total_whole} &nbsp;·&nbsp; {whole_pct}%</span>
            </div>
            <div class="class-row">
                <span><span class="class-dot" style="background:#ef4444"></span>broken_chip</span>
                <span style="color:#ef4444">{total_broken} &nbsp;·&nbsp; {broken_pct}%</span>
            </div>
            <div class="class-row">
                <span style="color:#888">⏱ Total processing time</span>
                <span style="color:#f5a623;font-family:'JetBrains Mono',monospace">{elapsed_total:.1f}s &nbsp;·&nbsp; {avg_fps:.1f} FPS</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-label" style="margin-top:16px">Export</div>', unsafe_allow_html=True)
            st.download_button(
                label     = "⬇ Download annotated video",
                data      = video_bytes,
                file_name = "chip_detection_output.mp4",
                mime      = "video/mp4"
            )

    else:
        st.info("👆 Upload a video file from the file uploader above, then click Run Detection.")


# ══════════════════════════════════════════════════════════════════════════════
# WEBCAM MODE
# ══════════════════════════════════════════════════════════════════════════════
else:  # 📸 Webcam
    st.markdown("""
    <div class="webcam-hint">
    📷 &nbsp;Point your camera at potato chips and click <strong>Take Photo</strong> to run defect detection.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">📸 Camera capture</div>', unsafe_allow_html=True)
    cam_image = st.camera_input(
        label="Take a photo",
        label_visibility="collapsed",
        key="webcam_capture"
    )

    # ── Run inference on captured frame ───────────────────────────────────────
    if cam_image is not None:
        file_bytes = np.asarray(bytearray(cam_image.read()), dtype=np.uint8)
        bgr_img    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if bgr_img is None:
            st.error("❌ Failed to decode webcam image. Try again.")
        else:
            st.markdown("---")
            st.markdown('<div class="section-label">Detection results</div>', unsafe_allow_html=True)

            with st.spinner("Running inference on webcam frame..."):
                t0 = time.time()
                annotated, whole_c, broken_c = run_inference(model, bgr_img, conf_thresh)
                elapsed = (time.time() - t0) * 1000

            render_results(bgr_img, annotated, whole_c, broken_c, elapsed)

    else:
        st.info("👆 Use the camera above to take a snapshot. Results will appear here automatically.")
