import streamlit as st
import pytesseract
import numpy as np
from PIL import Image
import pickle
from lime.lime_text import LimeTextExplainer

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.title("Sarcastic Meme Detector")

st.write("Upload a meme image to detect sarcasm")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

# Prediction function for LIME
def predict_proba(texts):
    vectors = vectorizer.transform(texts)
    preds = model.predict(vectors.toarray())

    preds = np.array(preds)

    if preds.ndim == 2 and preds.shape[1] == 1:
        preds = np.concatenate([1 - preds, preds], axis=1)

    return preds


# Text cleaning function
def clean_text(text):
    text = text.lower()
    return text


if uploaded_file is not None:

    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img = np.array(image)

    # OCR text extraction
    text = pytesseract.image_to_string(img, lang="ben")

    st.subheader("Extracted Text")
    st.write(text)

    cleaned_text = clean_text(text)

    # Convert to features
    text_vector = vectorizer.transform([cleaned_text])

    # Prediction
    prediction = model.predict(text_vector)

    if prediction[0] == 1:
        st.success("Prediction: Sarcastic")
    else:
        st.success("Prediction: Non-Sarcastic")


    # LIME explanation
    class_names = ["Non-Sarcastic", "Sarcastic"]
    explainer = LimeTextExplainer(class_names=class_names)

    exp = explainer.explain_instance(
        cleaned_text,
        predict_proba,
        num_features=10
    )

    st.subheader("Important Words (LIME Explanation)")
    st.write(exp.as_list())


    # Human readable explanation
    important_words = [word for word, score in exp.as_list()[:3]]

    st.subheader("Human Explanation")
    st.write(
        "The model focused on the words: "
        + ", ".join(important_words)
        + " which influenced the sarcasm prediction."
    )
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"