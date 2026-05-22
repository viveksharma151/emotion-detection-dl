import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import time
import base64

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmotiSense AI | Emotion Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* Hide default header */
header[data-testid="stHeader"] { background: transparent; }

/* Main title */
.hero-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    letter-spacing: -1px;
}
.hero-subtitle {
    text-align: center;
    color: rgba(255,255,255,0.55);
    font-size: 1.1rem;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* Cards */
.glass-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Emotion badge */
.emotion-badge {
    display: inline-block;
    padding: 0.5rem 1.5rem;
    border-radius: 50px;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin: 0.5rem 0;
}

/* Metric card */
.metric-box {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
    transition: transform 0.2s;
}
.metric-box:hover { transform: translateY(-4px); }
.metric-label { color: rgba(255,255,255,0.5); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { color: #fff; font-size: 1.8rem; font-weight: 700; margin-top: 0.3rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.6) !important;
}

/* Labels */
label, .stMarkdown p { color: rgba(255,255,255,0.8) !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border: 2px dashed rgba(124, 58, 237, 0.5);
    border-radius: 16px;
    padding: 1rem;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.1) !important; }

/* Alert boxes */
.stSuccess { background: rgba(52, 211, 153, 0.15) !important; border: 1px solid rgba(52,211,153,0.3) !important; border-radius: 12px !important; }
.stInfo { background: rgba(96, 165, 250, 0.15) !important; border: 1px solid rgba(96,165,250,0.3) !important; border-radius: 12px !important; }
.stWarning { background: rgba(251, 191, 36, 0.15) !important; border: 1px solid rgba(251,191,36,0.3) !important; border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Emotion Config ──────────────────────────────────────────────────────────
EMOTION_CONFIG = {
    "happy":     {"emoji": "😄", "color": "#fbbf24", "label": "Happy"},
    "sad":       {"emoji": "😢", "color": "#60a5fa", "label": "Sad"},
    "angry":     {"emoji": "😠", "color": "#f87171", "label": "Angry"},
    "surprise":  {"emoji": "😲", "color": "#a78bfa", "label": "Surprised"},
    "fear":      {"emoji": "😨", "color": "#94a3b8", "label": "Fearful"},
    "disgust":   {"emoji": "🤢", "color": "#34d399", "label": "Disgusted"},
    "neutral":   {"emoji": "😐", "color": "#e2e8f0", "label": "Neutral"},
}

# ─── Helper Functions ────────────────────────────────────────────────────────
def analyze_emotion(img_array):
    """Run DeepFace emotion analysis."""
    try:
        result = DeepFace.analyze(
            img_array,
            actions=["emotion", "age", "gender"],
            enforce_detection=False,
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]
        return result, None
    except Exception as e:
        return None, str(e)

def draw_faces(img_array, result):
    """Draw bounding box + label on the image."""
    img = img_array.copy()
    region = result.get("region", {})
    x, y, w, h = (
        region.get("x", 0), region.get("y", 0),
        region.get("w", 0), region.get("h", 0),
    )
    emotion = result.get("dominant_emotion", "neutral")
    cfg = EMOTION_CONFIG.get(emotion, EMOTION_CONFIG["neutral"])
    color_hex = cfg["color"].lstrip("#")
    r, g, b = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))

    if w > 0 and h > 0:
        cv2.rectangle(img, (x, y), (x + w, y + h), (r, g, b), 3)
        label = f"{cfg['emoji']} {cfg['label']}"
        cv2.rectangle(img, (x, y - 40), (x + w, y), (r, g, b), -1)
        cv2.putText(img, label, (x + 5, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    return img

def emotion_bar_chart(emotions: dict):
    """Plotly bar chart for emotion scores."""
    labels = [EMOTION_CONFIG.get(k, {}).get("label", k.capitalize()) for k in emotions.keys()]
    values = [round(v, 2) for v in emotions.values()]
    colors = [EMOTION_CONFIG.get(k, {}).get("color", "#7c3aed") for k in emotions.keys()]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.1)", width=1)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="white", size=12),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="white", family="Inter"),
        xaxis=dict(title="Confidence %", range=[0, 115], gridcolor="rgba(255,255,255,0.07)", color="rgba(255,255,255,0.5)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)", color="rgba(255,255,255,0.8)"),
        height=300,
        margin=dict(l=10, r=20, t=10, b=10),
        showlegend=False,
    )
    return fig

def emotion_gauge(dominant_score: float, emotion: str):
    """Plotly gauge for dominant emotion confidence."""
    cfg = EMOTION_CONFIG.get(emotion, EMOTION_CONFIG["neutral"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=dominant_score,
        title=dict(text=f"{cfg['emoji']} {cfg['label']}", font=dict(size=20, color="white")),
        number=dict(suffix="%", font=dict(color="white", size=28)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="white", tickfont=dict(color="rgba(255,255,255,0.5)")),
            bar=dict(color=cfg["color"]),
            bgcolor="rgba(255,255,255,0.05)",
            borderwidth=0,
            steps=[
                dict(range=[0, 33], color="rgba(255,255,255,0.04)"),
                dict(range=[33, 66], color="rgba(255,255,255,0.07)"),
                dict(range=[66, 100], color="rgba(255,255,255,0.1)"),
            ],
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig

# ─── Session State ───────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:3rem;'>🎭</div>
        <div style='font-weight:700; font-size:1.2rem; color:white;'>EmotiSense AI</div>
        <div style='color:rgba(255,255,255,0.45); font-size:0.8rem;'>Powered by DeepFace</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("**⚙️ Settings**")
    enforce_detection = st.toggle("Strict Face Detection", value=False,
                                   help="If ON, raises error when no face found. If OFF, still attempts analysis.")
    show_age_gender = st.toggle("Show Age & Gender", value=True)
    show_annotated  = st.toggle("Show Annotated Image", value=True)

    st.divider()
    st.markdown("**📊 Session Stats**")
    total = len(st.session_state.history)
    if total > 0:
        most_common = max(set(st.session_state.history), key=st.session_state.history.count)
        cfg = EMOTION_CONFIG.get(most_common, EMOTION_CONFIG["neutral"])
        st.markdown(f"""
        <div class='metric-box' style='margin-bottom:0.6rem;'>
            <div class='metric-label'>Total Detections</div>
            <div class='metric-value'>{total}</div>
        </div>
        <div class='metric-box'>
            <div class='metric-label'>Most Common</div>
            <div class='metric-value'>{cfg['emoji']} {cfg['label']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No detections yet.")

    if total > 0:
        st.divider()
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

    st.divider()
    st.markdown("""
    <div style='color:rgba(255,255,255,0.35); font-size:0.75rem; text-align:center;'>
    Built with DeepFace · TensorFlow · OpenCV<br>Streamlit · Plotly
    </div>
    """, unsafe_allow_html=True)

# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>🎭 EmotiSense AI</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Real-time Facial Emotion Detection powered by Deep Learning</div>", unsafe_allow_html=True)

# ─── Mode Tabs ───────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📁 Upload Image", "📷 Camera Capture", "📈 Emotion History"])

# ════════════════════════════════════════════════
# TAB 1 – Upload Image
# ════════════════════════════════════════════════
with tab1:
    col_up, col_res = st.columns([1, 1.2], gap="large")

    with col_up:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 📂 Upload a Face Image")
        uploaded = st.file_uploader(
            "Drag & drop or click to browse",
            type=["jpg", "jpeg", "png", "webp"],
            key="upload_img",
            label_visibility="collapsed",
        )

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            img_array = np.array(img)
            st.image(img, caption="Uploaded Image", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded and st.button("🔍 Detect Emotion", key="btn_upload"):
            with st.spinner("Analyzing emotions with DeepFace..."):
                result, err = analyze_emotion(img_array)

            if err:
                st.error(f"❌ Error: {err}")
            else:
                st.session_state["upload_result"] = (result, img_array)
                emotion = result.get("dominant_emotion", "neutral")
                st.session_state.history.append(emotion)
                st.success(f"✅ Analysis complete!")

    with col_res:
        if "upload_result" in st.session_state:
            result, img_array = st.session_state["upload_result"]
            emotion   = result.get("dominant_emotion", "neutral")
            emotions  = result.get("emotion", {})
            age       = result.get("age", "—")
            gender    = result.get("dominant_gender", "—")
            cfg       = EMOTION_CONFIG.get(emotion, EMOTION_CONFIG["neutral"])

            # Dominant emotion headline
            st.markdown(f"""
            <div class='glass-card' style='text-align:center; border-color:{cfg["color"]}40;'>
                <div style='font-size:4rem; margin-bottom:0.3rem;'>{cfg["emoji"]}</div>
                <div class='emotion-badge' style='background:{cfg["color"]}22; color:{cfg["color"]}; border:1.5px solid {cfg["color"]}55;'>
                    {cfg["label"]}
                </div>
                <div style='color:rgba(255,255,255,0.45); font-size:0.85rem; margin-top:0.5rem;'>
                    Dominant Emotion Detected
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Age / Gender metrics
            if show_age_gender:
                c1, c2 = st.columns(2)
                gender_emoji = "👨" if str(gender).lower() == "man" else "👩"
                c1.markdown(f"<div class='metric-box'><div class='metric-label'>Estimated Age</div><div class='metric-value'>~{age}</div></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-box'><div class='metric-label'>Gender</div><div class='metric-value'>{gender_emoji} {gender}</div></div>", unsafe_allow_html=True)

            # Gauge
            dom_score = emotions.get(emotion, 0)
            st.plotly_chart(emotion_gauge(dom_score, emotion), use_container_width=True)

            # Bar chart
            st.markdown("**Emotion Probability Distribution**")
            st.plotly_chart(emotion_bar_chart(emotions), use_container_width=True)

            # Annotated image
            if show_annotated:
                annotated = draw_faces(img_array, result)
                st.image(annotated, caption="Annotated Output", use_container_width=True)
        else:
            st.markdown("""
            <div class='glass-card' style='text-align:center; padding:3rem 1rem;'>
                <div style='font-size:4rem;'>🧠</div>
                <div style='color:rgba(255,255,255,0.45); margin-top:0.8rem;'>
                    Upload an image and click<br><strong style='color:rgba(255,255,255,0.7);'>Detect Emotion</strong> to see results
                </div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════
# TAB 2 – Camera Capture
# ════════════════════════════════════════════════
with tab2:
    col_cam, col_cam_res = st.columns([1, 1.2], gap="large")

    with col_cam:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 📷 Capture from Camera")
        st.markdown("<small style='color:rgba(255,255,255,0.45);'>Allow camera access when prompted</small>", unsafe_allow_html=True)
        camera_photo = st.camera_input("Take a photo", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        if camera_photo and st.button("🔍 Detect Emotion", key="btn_camera"):
            img = Image.open(camera_photo).convert("RGB")
            img_array = np.array(img)
            with st.spinner("Analyzing emotions with DeepFace..."):
                result, err = analyze_emotion(img_array)

            if err:
                st.error(f"❌ Error: {err}")
            else:
                st.session_state["camera_result"] = (result, img_array)
                emotion = result.get("dominant_emotion", "neutral")
                st.session_state.history.append(emotion)
                st.success("✅ Analysis complete!")

    with col_cam_res:
        if "camera_result" in st.session_state:
            result, img_array = st.session_state["camera_result"]
            emotion  = result.get("dominant_emotion", "neutral")
            emotions = result.get("emotion", {})
            age      = result.get("age", "—")
            gender   = result.get("dominant_gender", "—")
            cfg      = EMOTION_CONFIG.get(emotion, EMOTION_CONFIG["neutral"])

            st.markdown(f"""
            <div class='glass-card' style='text-align:center; border-color:{cfg["color"]}40;'>
                <div style='font-size:4rem; margin-bottom:0.3rem;'>{cfg["emoji"]}</div>
                <div class='emotion-badge' style='background:{cfg["color"]}22; color:{cfg["color"]}; border:1.5px solid {cfg["color"]}55;'>
                    {cfg["label"]}
                </div>
                <div style='color:rgba(255,255,255,0.45); font-size:0.85rem; margin-top:0.5rem;'>
                    Dominant Emotion Detected
                </div>
            </div>
            """, unsafe_allow_html=True)

            if show_age_gender:
                c1, c2 = st.columns(2)
                gender_emoji = "👨" if str(gender).lower() == "man" else "👩"
                c1.markdown(f"<div class='metric-box'><div class='metric-label'>Estimated Age</div><div class='metric-value'>~{age}</div></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-box'><div class='metric-label'>Gender</div><div class='metric-value'>{gender_emoji} {gender}</div></div>", unsafe_allow_html=True)

            dom_score = emotions.get(emotion, 0)
            st.plotly_chart(emotion_gauge(dom_score, emotion), use_container_width=True)
            st.markdown("**Emotion Probability Distribution**")
            st.plotly_chart(emotion_bar_chart(emotions), use_container_width=True)

            if show_annotated:
                annotated = draw_faces(img_array, result)
                st.image(annotated, caption="Annotated Output", use_container_width=True)
        else:
            st.markdown("""
            <div class='glass-card' style='text-align:center; padding:3rem 1rem;'>
                <div style='font-size:4rem;'>📸</div>
                <div style='color:rgba(255,255,255,0.45); margin-top:0.8rem;'>
                    Take a photo and click<br><strong style='color:rgba(255,255,255,0.7);'>Detect Emotion</strong> to see results
                </div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════
# TAB 3 – Emotion History
# ════════════════════════════════════════════════
with tab3:
    if not st.session_state.history:
        st.markdown("""
        <div class='glass-card' style='text-align:center; padding:3rem;'>
            <div style='font-size:3rem;'>📊</div>
            <div style='color:rgba(255,255,255,0.45); margin-top:0.8rem;'>
                No history yet. Analyze some images to see trends!
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        history = st.session_state.history
        counts = {e: history.count(e) for e in set(history)}
        labels = [EMOTION_CONFIG.get(k, {}).get("label", k) for k in counts.keys()]
        emojis = [EMOTION_CONFIG.get(k, {}).get("emoji", "") for k in counts.keys()]
        colors = [EMOTION_CONFIG.get(k, {}).get("color", "#7c3aed") for k in counts.keys()]
        values = list(counts.values())

        col_h1, col_h2 = st.columns([1, 1], gap="large")

        with col_h1:
            st.markdown("#### 🥧 Emotion Distribution")
            pie_labels = [f"{e} {l}" for e, l in zip(emojis, labels)]
            fig_pie = go.Figure(go.Pie(
                labels=pie_labels, values=values,
                marker=dict(colors=colors, line=dict(color="#0f0c29", width=2)),
                hole=0.5,
                textinfo="label+percent",
                textfont=dict(size=13, color="white"),
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="Inter"),
                showlegend=False,
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_h2:
            st.markdown("#### 📋 Timeline")
            for i, e in enumerate(reversed(history[-10:]), 1):
                cfg = EMOTION_CONFIG.get(e, EMOTION_CONFIG["neutral"])
                st.markdown(f"""
                <div style='display:flex; align-items:center; gap:0.8rem;
                     background:rgba(255,255,255,0.05); border-radius:10px;
                     padding:0.5rem 0.8rem; margin-bottom:0.4rem;
                     border-left: 3px solid {cfg["color"]};'>
                    <span style='font-size:1.4rem;'>{cfg["emoji"]}</span>
                    <span style='color:white; font-weight:600;'>{cfg["label"]}</span>
                    <span style='color:rgba(255,255,255,0.35); font-size:0.8rem; margin-left:auto;'>#{len(history)-i+1}</span>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 📈 Emotion Frequency Bar")
        fig_bar2 = go.Figure(go.Bar(
            x=[f"{e} {l}" for e, l in zip(emojis, labels)],
            y=values,
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.1)", width=1)),
            text=values, textposition="outside",
            textfont=dict(color="white"),
        ))
        fig_bar2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.03)",
            font=dict(color="white", family="Inter"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.07)", color="rgba(255,255,255,0.6)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.07)", color="rgba(255,255,255,0.6)", title="Count"),
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_bar2, use_container_width=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:rgba(255,255,255,0.08); margin-top:2rem;'>
<div style='text-align:center; color:rgba(255,255,255,0.3); font-size:0.8rem; padding-bottom:1rem;'>
    🎭 EmotiSense AI &nbsp;·&nbsp; Built with ❤️ using DeepFace, TensorFlow & Streamlit<br>
    <span style='font-size:0.7rem;'>Deep Learning · Computer Vision · Real-time Analysis</span>
</div>
""", unsafe_allow_html=True)
