import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_translator import GoogleTranslator
import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os

# -------- CONFIG --------
API_KEY = "Enter your api key here"
genai.configure(api_key=API_KEY)
llm_model = genai.GenerativeModel('gemini-1.5-flash')

LANGUAGES = {
    'English': 'en',
    'Hindi': 'hi',
    'Telugu': 'te',
    'Tamil': 'ta',
    'Bengali': 'bn'
}

@st.cache_resource(show_spinner=False)
def load_yolo():
    # Load a pretrained YOLO model (replace with your medical weights if available)
    model = YOLO("yolov5n.pt")  # or "yolov7.pt" if you have it locally
    return model

def detect_objects_yolo(model, img):
    results = model(img)  # inference, returns a list of Results (one per image)
    labels_map = model.names  # class id -> label mapping
    detections = []

    # We process first image's results
    result = results[0]

    # Boxes tensor: shape (n,6): x1,y1,x2,y2,confidence,class
    boxes = result.boxes.data.cpu().numpy()

    for *box, conf, cls in boxes:
        label = labels_map[int(cls)]
        detections.append({
            'label': label,
            'confidence': float(conf),
            'box': box
        })

    return detections

def generate_insight(query, lang="en"):
    prompt = f"Provide medical insights for: {query}. Include causes, advice, and when to consult a doctor."
    response = llm_model.generate_content(prompt)
    text = response.text
    if lang != "en":
        text = GoogleTranslator(source='auto', target=lang).translate(text)
    return text

def play_audio(text, lang='en'):
    tts = gTTS(text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format='audio/mp3')
    os.remove(fp.name)

def record_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio_data = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio_data)
        except Exception as e:
            return "Sorry, I couldn't understand that."

# -------- UI --------
st.set_page_config(page_title="AI Medical Diagnosis Chatbot with YOLO", layout="wide")
st.markdown("<h1 style='text-align:center;'>🩺 AI Medical Diagnosis Chatbot (YOLO)</h1>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    image = st.file_uploader("📤 Upload Medical Image (X-ray, CT, MRI, Eye images, etc.)",
                             type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"])
    lang_choice = st.selectbox("🌍 Choose Language for Response", list(LANGUAGES.keys()), index=0)
    tts_enabled = st.checkbox("🔊 Enable Voice Output")
    text_query = st.text_input("💬 Ask any medical question (without image)")
    if st.button("🎤 Speak Question"):
        user_voice_input = record_speech()
        st.success(f"🗣 You said: {user_voice_input}")
        text_query = user_voice_input

with col2:
    st.image("https://th.bing.com/th/id/OIP.gxF3ONyjgXMC-cwFkhjamgAAAA?w=153&h=180&c=7&r=0&o=7&dpr=2&pid=1.7&rm=3",
             use_container_width=True)

# Load model once
yolo_model = load_yolo()

# Handle text query only
if text_query:
    st.markdown("### 🤖 AI Response:")
    insight = generate_insight(text_query, lang=LANGUAGES[lang_choice])
    st.success(insight)
    if tts_enabled:
        play_audio(insight, lang=LANGUAGES[lang_choice])

# Handle image detection
if image:
    st.markdown("### 🧠 Analyzing Image with YOLO...")

    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    detections = detect_objects_yolo(yolo_model, img)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if detections:
        st.markdown("### 🔍 Detected Objects:")
        for det in detections:
            label = det['label']
            conf = det['confidence']
            st.write(f"- {label} (Confidence: {conf:.2f})")

        for det in detections:
            label = det['label']
            explanation = generate_insight(label, lang=LANGUAGES[lang_choice])
            st.markdown(f"#### 📌 Explanation for: *{label}*")
            st.info(explanation)
            if tts_enabled:
                play_audio(explanation, lang=LANGUAGES[lang_choice])
    else:
        st.warning("No objects detected. Note: Default YOLO models are not trained specifically on medical data.")

st.markdown("<hr><center>Made with ❤ by Rahul Kumar</center>", unsafe_allow_html=True)
