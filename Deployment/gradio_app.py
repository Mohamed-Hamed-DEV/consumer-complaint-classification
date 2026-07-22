"""
Gradio deployment app for the Consumer Complaint Classification project.

Usage:
1. Download the `best_model` folder from Kaggle's Output tab after running
   the notebook, and place it next to this script (or update MODEL_DIR below).
2. pip install -r requirements.txt   (see bottom of this file for the list)
3. python gradio_app.py
4. Open the local URL Gradio prints (usually http://127.0.0.1:7860)

This script auto-detects whether the saved best model is one of the Keras
RNN models (SimpleRNN/LSTM/GRU) or the fine-tuned HuggingFace Transformer,
based on the `model_type.txt` file saved by the notebook.
"""

import os
import re
import pickle
import numpy as np
import gradio as gr

MODEL_DIR = "best_model"  # update if your folder is named/located differently

# ---- preprocessing (must match the notebook's clean_text exactly) ----
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# ---- load model type ----
with open(os.path.join(MODEL_DIR, "model_type.txt")) as f:
    MODEL_TYPE = f.read().strip()

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

if MODEL_TYPE == "transformer":
    import tensorflow as tf
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    def predict(text):
        cleaned = clean_text(text)
        inputs = tokenizer(cleaned, return_tensors="tf", truncation=True, max_length=150)
        logits = model(**inputs).logits
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]
        idx = int(np.argmax(probs))
        label = label_encoder.inverse_transform([idx])[0]
        confidence = float(probs[idx])
        return label, confidence

else:  # keras_rnn
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "model.keras"))

    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "max_len.txt")) as f:
        MAX_LEN = int(f.read().strip())

    def predict(text):
        cleaned = clean_text(text)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        probs = model.predict(padded, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = label_encoder.inverse_transform([idx])[0]
        confidence = float(probs[idx])
        return label, confidence


def classify_complaint(complaint_text):
    if not complaint_text or not complaint_text.strip():
        return "Please enter a complaint.", ""
    label, confidence = predict(complaint_text)
    return label, f"{confidence * 100:.1f}%"


demo = gr.Interface(
    fn=classify_complaint,
    inputs=gr.Textbox(
        lines=6,
        label="Customer Complaint",
        placeholder="Enter the complaint narrative here..."
    ),
    outputs=[
        gr.Textbox(label="Predicted Category"),
        gr.Textbox(label="Confidence Score"),
    ],
    title="Consumer Complaint Classifier",
    description=(
        f"Model type: {MODEL_TYPE}. Enter a customer complaint below and the model "
        "will predict its category with a confidence score."
    ),
    examples=[
        ["I have been trying to dispute an incorrect item on my credit report for months and no one responds."],
        ["My mortgage servicer keeps charging me late fees even though I paid on time."],
        ["A debt collector keeps calling me multiple times a day about a debt that isn't mine."],
    ],
)

if __name__ == "__main__":
    demo.launch()
