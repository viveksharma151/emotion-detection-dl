import streamlit as st
import cv2
import numpy as np
from transformers import pipeline
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# color scheme and emoji for each emotion label
# keeping this separate so i can change it easily
EMOTIONS = {
    "happy":    {"emoji": "😄", "color": "#fbbf24", "label": "Happy"},
    "sad":      {"emoji": "😢", "color": "#60a5fa", "label": "Sad"},
    "angry":    {"emoji": "😠", "color": "#f87171", "label": "Angry"},
    "surprise": {"emoji": "😲", "color": "#a78bfa", "label": "Surprised"},
    "fear":     {"emoji": "😨", "color": "#94a3b8", "label": "Fearful"},
    "disgust":  {"emoji": "🤢", "color": "#34d399", "label": "Disgusted"},
    "neutral":  {"emoji": "😐", "color": "#e2e8f0", "label": "Neutral"},
}

# the huggingface model uses slightly different names in some cases
# this maps them to our keys above
LABEL_MAP = {
    "happy":     "happy",
    "sad":       "sad",
    "angry":     "angry",
    "surprise":  "surprise",
    "surprised": "surprise",
    "fear":      "fear",
    "fearful":   "fear",
    "disgust":   "disgust",
    "disgusted": "disgust",
    "neutral":   "neutral",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}

header[data-testid="stHeader"] {
    background: transparent;
}

.page-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    margin-bottom: 0.2rem;
}

.page-sub {
    text-align: center;
    color: rgba(255,255,255,0.5);
    font-size: 1rem;
    margin-bottom: 2rem;
}

.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 18px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}

.emotion-tag {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    font-size: 1.3rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stat-box {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
    margin-bottom: 0.6rem;
}
.stat-label { color: rgba(255,255,255,0.45); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }
.stat-val   { color: white; font-size: 1.7rem; font-weight: 700; margin-top: 0.2rem; }

[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.03) !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.4rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    box-shadow: 0 4px 14px rgba(124,58,237,0.35) !important;
}

label, .stMarkdown p { color: rgba(255,255,255,0.8) !important; }
hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)


# load model once and cache it — took me a while to figure out cache_resource is the right one here
# cache_data doesn't work with pipeline objects
@st.cache_resource(show_spinner=False)
def load_emotion_model():
    return pipeline(
        "image-classification",
        model="trpakov/vit-face-expression",
        top_k=7,
    )


def run_inference(pil_img):
    """
    runs the vit model on a PIL image
    returns (result_dict, error_string)
    result_dict has 'dominant_emotion' and 'emotion' (dict of scores)
    """
    try:
        model = load_emotion_model()
        raw = model(pil_img)

        scores = {}
        for item in raw:
            key = LABEL_MAP.get(item["label"].lower(), item["label"].lower())
            scores[key] = round(item["score"] * 100, 2)

        dominant = max(scores, key=scores.get)
        return {"dominant_emotion": dominant, "emotion": scores}, None

    except Exception as e:
        return None, str(e)


def make_bar_chart(scores: dict):
    # sort by score so highest is at top
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    ys = [EMOTIONS.get(k, {}).get("label", k.title()) for k, _ in sorted_scores]
    xs = [v for _, v in sorted_scores]
    clrs = [EMOTIONS.get(k, {}).get("color", "#7c3aed") for k, _ in sorted_scores]

    fig = go.Figure(go.Bar(
        x=xs, y=ys,
        orientation="h",
        marker=dict(color=clrs),
        text=[f"{v:.1f}%" for v in xs],
        textposition="outside",
        textfont=dict(color="white", size=12),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="white", family="Inter"),
        xaxis=dict(range=[0, 115], gridcolor="rgba(255,255,255,0.06)", color="rgba(255,255,255,0.4)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", color="rgba(255,255,255,0.7)"),
        height=300,
        margin=dict(l=5, r=20, t=5, b=5),
        showlegend=False,
    )
    return fig


def make_gauge(score, emotion):
    cfg = EMOTIONS.get(emotion, EMOTIONS["neutral"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title=dict(text=f"{cfg['emoji']}  {cfg['label']}", font=dict(size=18, color="white")),
        number=dict(suffix="%", font=dict(color="white", size=26)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color="rgba(255,255,255,0.4)")),
            bar=dict(color=cfg["color"]),
            bgcolor="rgba(255,255,255,0.04)",
            borderwidth=0,
            steps=[
                dict(range=[0, 33],  color="rgba(255,255,255,0.03)"),
                dict(range=[33, 66], color="rgba(255,255,255,0.05)"),
                dict(range=[66, 100],color="rgba(255,255,255,0.08)"),
            ],
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        height=210,
        margin=dict(l=20, r=20, t=35, b=5),
    )
    return fig


def show_result(result):
    """renders the emotion result — dominant emotion + gauge + bar chart"""
    emotion = result["dominant_emotion"]
    scores  = result["emotion"]
    cfg     = EMOTIONS.get(emotion, EMOTIONS["neutral"])
    top_score = scores.get(emotion, 0)

    st.markdown(f"""
    <div class='card' style='text-align:center; border-color:{cfg["color"]}33;'>
        <div style='font-size:3.5rem;'>{cfg["emoji"]}</div>
        <div class='emotion-tag' style='background:{cfg["color"]}18; color:{cfg["color"]}; border:1px solid {cfg["color"]}44;'>
            {cfg["label"]}
        </div>
        <div style='color:rgba(255,255,255,0.4); font-size:0.82rem; margin-top:0.5rem;'>
            {top_score:.1f}% confidence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(make_gauge(top_score, emotion), use_container_width=True)
    st.markdown("**All emotion scores**")
    st.plotly_chart(make_bar_chart(scores), use_container_width=True)


# session state for history
if "history" not in st.session_state:
    st.session_state.history = []


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 0.5rem;'>
        <div style='font-size:2.5rem;'>🎭</div>
        <div style='color:white; font-weight:700; font-size:1.1rem;'>Emotion Detector</div>
        <div style='color:rgba(255,255,255,0.4); font-size:0.75rem;'>ViT · HuggingFace</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # load the model on startup so first detection is faster
    with st.spinner("Loading model..."):
        load_emotion_model()
    st.success("Model loaded ✓")

    st.divider()

    total = len(st.session_state.history)
    if total > 0:
        top_emotion = max(set(st.session_state.history), key=st.session_state.history.count)
        ecfg = EMOTIONS.get(top_emotion, EMOTIONS["neutral"])
        st.markdown(f"""
        <div class='stat-box'>
            <div class='stat-label'>Detections</div>
            <div class='stat-val'>{total}</div>
        </div>
        <div class='stat-box'>
            <div class='stat-label'>Most common</div>
            <div class='stat-val'>{ecfg['emoji']} {ecfg['label']}</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No detections yet")


# ── main content ─────────────────────────────────────────────────────────────
st.markdown("<div class='page-title'>🎭 Emotion Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='page-sub'>Upload a face image or use your camera — model predicts the emotion</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Upload Image", "Camera", "History"])

# upload tab
with tab1:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        uploaded = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, use_container_width=True)

        if uploaded and st.button("Detect emotion", key="btn_upload"):
            with st.spinner("Running inference..."):
                result, err = run_inference(img)
            if err:
                st.error(f"Something went wrong: {err}")
            else:
                st.session_state["upload_result"] = result
                st.session_state.history.append(result["dominant_emotion"])

    with right:
        if "upload_result" in st.session_state:
            show_result(st.session_state["upload_result"])
        else:
            st.markdown("""
            <div class='card' style='text-align:center; padding:2.5rem 1rem;'>
                <div style='font-size:3rem;'>🧠</div>
                <div style='color:rgba(255,255,255,0.4); margin-top:0.6rem; font-size:0.9rem;'>
                    Upload an image and hit detect
                </div>
            </div>
            """, unsafe_allow_html=True)

# camera tab
with tab2:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        photo = st.camera_input("Take a photo")

        if photo and st.button("Detect emotion", key="btn_cam"):
            img = Image.open(photo).convert("RGB")
            with st.spinner("Running inference..."):
                result, err = run_inference(img)
            if err:
                st.error(f"Something went wrong: {err}")
            else:
                st.session_state["cam_result"] = result
                st.session_state.history.append(result["dominant_emotion"])

    with right:
        if "cam_result" in st.session_state:
            show_result(st.session_state["cam_result"])
        else:
            st.markdown("""
            <div class='card' style='text-align:center; padding:2.5rem 1rem;'>
                <div style='font-size:3rem;'>📸</div>
                <div style='color:rgba(255,255,255,0.4); margin-top:0.6rem; font-size:0.9rem;'>
                    Take a photo and hit detect
                </div>
            </div>
            """, unsafe_allow_html=True)

# history tab
with tab3:
    history = st.session_state.history

    if not history:
        st.markdown("""
        <div class='card' style='text-align:center; padding:2.5rem;'>
            <div style='font-size:2.5rem;'>📊</div>
            <div style='color:rgba(255,255,255,0.4); margin-top:0.6rem; font-size:0.9rem;'>
                Nothing here yet — analyze some images first
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        counts  = {e: history.count(e) for e in set(history)}
        labels  = [EMOTIONS.get(k, {}).get("label", k) for k in counts]
        emojis  = [EMOTIONS.get(k, {}).get("emoji", "") for k in counts]
        colors  = [EMOTIONS.get(k, {}).get("color", "#7c3aed") for k in counts]
        values  = list(counts.values())

        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown("**Distribution**")
            fig_pie = go.Figure(go.Pie(
                labels=[f"{e} {l}" for e, l in zip(emojis, labels)],
                values=values,
                marker=dict(colors=colors, line=dict(color="#0f0c29", width=2)),
                hole=0.45,
                textfont=dict(size=12, color="white"),
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="Inter"),
                showlegend=False,
                height=280,
                margin=dict(l=5, r=5, t=5, b=5),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown("**Last 10 detections**")
            for i, e in enumerate(reversed(history[-10:]), 1):
                ecfg = EMOTIONS.get(e, EMOTIONS["neutral"])
                st.markdown(f"""
                <div style='display:flex; align-items:center; gap:0.7rem;
                     background:rgba(255,255,255,0.04);
                     border-left:3px solid {ecfg["color"]};
                     border-radius:8px; padding:0.45rem 0.8rem;
                     margin-bottom:0.35rem;'>
                    <span style='font-size:1.2rem;'>{ecfg["emoji"]}</span>
                    <span style='color:white; font-weight:500;'>{ecfg["label"]}</span>
                    <span style='margin-left:auto; color:rgba(255,255,255,0.3); font-size:0.78rem;'>#{len(history)-i+1}</span>
                </div>
                """, unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; color:rgba(255,255,255,0.2); font-size:0.75rem; margin-top:2rem; padding-bottom:1rem;'>
    built with streamlit · huggingface transformers · plotly
</div>
""", unsafe_allow_html=True)
