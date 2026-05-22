# Emotion Detection using Vision Transformer

A deep learning project that detects facial emotions from images. Built this to learn about vision transformers and how HuggingFace models work in practice.

Live demo → **[emotion-detection-dl-tofcykpkcxe3ler7bauoaj.streamlit.app](https://emotion-detection-dl-tofcykpkcxe3ler7bauoaj.streamlit.app)**

---

## What it does

Upload a photo or use your webcam, and the model will tell you what emotion is on the face — happy, sad, angry, surprised, fearful, disgusted, or neutral. Also shows confidence scores for each emotion with some charts.

Tried DeepFace first but it was a pain to deploy (TensorFlow compatibility issues), so switched to HuggingFace's ViT model which is cleaner and works everywhere.

---

## Tech used

- **Python** — main language
- **Streamlit** — for the web UI, super quick to get things running
- **HuggingFace Transformers** — `trpakov/vit-face-expression` model for emotion classification
- **PyTorch** — backend for running the ViT model
- **OpenCV** — image processing
- **Plotly** — for the charts (gauge + bar chart)

---

## Emotions it can detect

| Emotion | |
|---|---|
| Happy | 😄 |
| Sad | 😢 |
| Angry | 😠 |
| Surprised | 😲 |
| Fearful | 😨 |
| Disgusted | 🤢 |
| Neutral | 😐 |

---

## Running locally

Clone and install dependencies:

```bash
git clone https://github.com/viveksharma151/emotion-detection-dl.git
cd emotion-detection-dl
```

Create a virtual environment (recommended):

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

Install packages:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

First time running will download the ViT model weights (~350MB), so give it a minute.

---

## Project structure

```
emotion-detection-dl/
├── app.py              # main streamlit app
├── requirements.txt    # dependencies
├── .gitignore
└── README.md
```

---

## How the model works

The model is `trpakov/vit-face-expression`, a Vision Transformer (ViT) fine-tuned on facial expression datasets. ViT splits the image into patches, processes them like tokens (similar to how BERT handles words), and classifies the emotion.

It's more accurate than CNN-based approaches for this task since attention mechanisms are better at capturing subtle facial features like slight brow raises or lip movements.

---

## Screenshots

Upload tab — upload any face image and get emotion scores instantly.

Camera tab — capture directly from webcam.

History tab — tracks all your detections in the session with a pie chart breakdown.

---

## Known issues / limitations

- Works best with clear, front-facing photos
- Multiple faces in one image — it picks up the dominant one
- Poor lighting can affect accuracy
- First load is slow because of model download (cached after that)

---

## What I learned

- How Vision Transformers work compared to CNNs
- Using HuggingFace `pipeline` API for image classification
- Streamlit session state for tracking history across interactions
- Plotly for interactive charts in Streamlit

---

## Contact

Vivek Sharma — [GitHub](https://github.com/viveksharma151)
