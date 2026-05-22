# 🎭 EmotiSense AI — Facial Emotion Detection

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![DeepFace](https://img.shields.io/badge/DeepFace-0.0.93-purple?style=for-the-badge)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> **Real-time facial emotion detection** powered by Deep Learning. Detects 7 emotions from face images with confidence scores, bounding boxes, age & gender estimation — all in a stunning dark-mode web UI.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎭 **7 Emotions** | Happy, Sad, Angry, Surprised, Fearful, Disgusted, Neutral |
| 📁 **Image Upload** | Upload JPG / PNG / WebP images for analysis |
| 📷 **Camera Capture** | Use your webcam to capture and analyze in real-time |
| 📊 **Confidence Gauge** | Plotly gauge chart showing dominant emotion confidence |
| 📈 **Distribution Chart** | Full probability bar chart for all 7 emotions |
| 🧑 **Age & Gender** | Estimates subject's age and gender using DeepFace |
| 🖼️ **Annotated Output** | Face bounding box with emotion label drawn on image |
| 📉 **Session History** | Tracks all detections and shows trends & pie chart |
| 🌑 **Dark Mode UI** | Premium glassmorphism design with gradient accents |

---

## 🧠 How It Works

```
Input Image / Camera Frame
        ↓
  OpenCV Face Detection
        ↓
  DeepFace Deep Learning Models
  (VGG-Face / FaceNet / ArcFace)
        ↓
  Emotion Classification (7 classes)
  Age Regression
  Gender Classification
        ↓
  Plotly Visualizations + Streamlit UI
```

DeepFace is a lightweight framework that wraps state-of-the-art face recognition models including **VGG-Face**, **FaceNet**, **OpenFace**, **DeepFace**, **DeepID**, **ArcFace**, and **Dlib**.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/emotion-detection-dl.git
cd emotion-detection-dl
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` 🎉

---

## 📦 Dependencies

```
streamlit          ≥ 1.32.0    — Web UI framework
deepface           ≥ 0.0.93    — Facial analysis (emotion, age, gender)
tensorflow         ≥ 2.13.0    — Deep learning backend
opencv-python-headless          — Image processing
plotly             ≥ 5.18.0    — Interactive charts
Pillow             ≥ 10.2.0    — Image handling
numpy              ≥ 1.24.0    — Numerical computing
```

> ⚠️ **Note**: First run will automatically download DeepFace model weights (~500MB). Subsequent runs are fast.

---

## 🗂️ Project Structure

```
emotion-detection-dl/
│
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

---

## 📸 Screenshots

> Upload a face photo → Instant emotion detection with confidence visualization

**Emotion Detection Result:**
- 🎭 Dominant emotion badge with color coding
- 📊 Plotly gauge (0–100% confidence)
- 📈 Full distribution bar chart (all 7 emotions)
- 🖼️ Annotated image with bounding box
- 🧑 Age (~25) & Gender (Male/Female)

**Session History Tab:**
- 🥧 Pie chart of emotion distribution
- 📋 Timeline of last 10 detections
- 📈 Frequency bar chart

---

## 🎯 Emotion Classes

| Emotion | Emoji | Description |
|---|---|---|
| Happy | 😄 | Joyful, smiling expression |
| Sad | 😢 | Downturned expression, tearful |
| Angry | 😠 | Furrowed brow, tense expression |
| Surprise | 😲 | Wide eyes, open mouth |
| Fear | 😨 | Raised eyebrows, tense |
| Disgust | 🤢 | Nose wrinkle, lip curl |
| Neutral | 😐 | Relaxed, expressionless |

---

## 🌐 Deployment

### Deploy on Streamlit Cloud (Free)
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Click **Deploy** 🚀

---

## 🛠️ Tech Stack

- **Deep Learning**: DeepFace, TensorFlow, Keras
- **Computer Vision**: OpenCV
- **Web Framework**: Streamlit
- **Visualization**: Plotly
- **Language**: Python 3.9+

---

## 👨‍💻 Author

**Vivek Sharma**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

<div align="center">
  <strong>⭐ If you found this useful, please star the repo!</strong><br>
  Built with ❤️ using Deep Learning & Streamlit
</div>
