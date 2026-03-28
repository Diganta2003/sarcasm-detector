import streamlit as st
import numpy as np
from PIL import Image
import pickle
import pytesseract
import tensorflow as tf
from lime.lime_text import LimeTextExplainer
import re
import pandas as pd

import os
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

@st.cache_resource
def load_models():
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    model = tf.keras.models.load_model("sarcasm_model.h5")
    return vectorizer, model

vectorizer, model = load_models()

@st.cache_resource
def load_tokenizer():
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.word_index = {v: k for k, v in enumerate(vectorizer.vocabulary_)}
    return tokenizer

tokenizer = load_tokenizer()

def cnn_predict_proba(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    padded = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=100)
    prob_sarc = model.predict(padded, verbose=0).flatten()
    return np.column_stack([1 - prob_sarc, prob_sarc])

def clean_text(text):
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text

st.set_page_config(page_title="Bengali Sarcasm Detector", page_icon="🎭", layout="centered")
st.title("🎭 Bengali Sarcastic Meme Detector")
st.write("Upload a Bengali meme image to detect whether it is sarcastic or not.")

uploaded_file = st.file_uploader("📷 Upload Meme Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Meme", use_column_width=True)

    with st.spinner("🔍 Extracting Bengali text from image..."):
        try:
            text = pytesseract.image_to_string(image, lang="ben")
        except Exception as e:
            st.error(f"OCR Error: {e}")
            st.stop()

    st.subheader("📝 Extracted Text")
    if text.strip():
        st.write(text)
    else:
        st.warning("⚠️ No Bengali text detected. Try a clearer image.")
        st.stop()

    cleaned = clean_text(text)

    with st.spinner("🤖 Predicting..."):
        seqs = tokenizer.texts_to_sequences([cleaned])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=100)
        prob = model.predict(padded, verbose=0).flatten()[0]
        prediction = 1 if prob >= 0.5 else 0

    st.subheader("🎯 Prediction Result")
    if prediction == 1:
        st.error("🔴 This meme is **SARCASTIC**")
    else:
        st.success("🟢 This meme is **NOT Sarcastic**")

    st.write(f"Confidence — Non-Sarcastic: `{(1-prob)*100:.1f}%` | Sarcastic: `{prob*100:.1f}%`")

    with st.spinner("🔎 Generating LIME explanation..."):
        try:
            explainer = LimeTextExplainer(
                class_names=["Non-Sarcastic", "Sarcastic"],
                split_expression=r'\s+',
                bow=True
            )
            exp = explainer.explain_instance(
                cleaned,
                cnn_predict_proba,
                num_features=10,
                num_samples=200
            )

            st.subheader("💡 LIME Explanation — Important Words")
            lime_list = exp.as_list()

            if lime_list:
                lime_df = pd.DataFrame(lime_list, columns=["Word", "Importance"])
                lime_df["Signal"] = lime_df["Importance"].apply(
                    lambda x: "🔴 Sarcastic" if x > 0 else "🔵 Non-Sarcastic"
                )
                st.dataframe(lime_df, use_container_width=True)

                top_words = [w for w, s in lime_list[:3]]
                st.subheader("🗣️ Human Explanation")
                st.info(
                    f"The model focused on: **{', '.join(top_words)}** "
                    f"which most influenced the prediction."
                )
        except Exception as e:
            st.warning(f"LIME explanation failed: {e}")

else:
    st.info("👆 Please upload a Bengali meme image to get started.")