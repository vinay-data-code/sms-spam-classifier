"""
SMS Spam Classifier - Streamlit App
-------------------------------------
To run (inside this folder):
    streamlit run app.py

Before running for the first time, run train_model.py so that model.pkl
and vectorizer.pkl get created (if they don't already exist).
"""

import re
import string
import pickle

import streamlit as st

STOPWORDS = set("""
i me my myself we our ours ourselves you you're you've you'll you'd your yours
yourself yourselves he him his himself she she's her hers herself it it's its
itself they them their theirs themselves what which who whom this that that'll
these those am is are was were be been being have has had having do does did
doing a an the and but if or because as until while of at by for with about
against between into through during before after above below to from up down
in out on off over under again further then once here there when where why
how all any both each few more most other some such no nor not only own same
so than too very s t can will just don don't should now
""".split())


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(words)


@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="centered")

st.title("📩 SMS Spam Classifier")
st.write("A Machine Learning model (TF-IDF + Naive Bayes) that predicts whether an SMS is **Spam** or **Ham (Not Spam)**.")

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error("model.pkl / vectorizer.pkl not found. Please run `python train_model.py` in the terminal first.")
    st.stop()

message = st.text_area("Enter or paste an SMS / message below:", height=150, placeholder="e.g. Congratulations! You have won a free prize, click here to claim now...")

col1, col2 = st.columns([1, 1])
with col1:
    predict_btn = st.button("🔍 Predict", use_container_width=True)
with col2:
    clear_btn = st.button("🧹 Clear", use_container_width=True)

if clear_btn:
    st.rerun()

if predict_btn:
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        cleaned = clean_text(message)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        spam_prob = proba[1] * 100
        ham_prob = proba[0] * 100

        if pred == 1:
            st.error(f"🚨 This message is **SPAM**! (confidence: {spam_prob:.1f}%)")
        else:
            st.success(f"✅ This message is **HAM (Not Spam)**. (confidence: {ham_prob:.1f}%)")

        st.progress(int(spam_prob))
        st.caption(f"Spam probability: {spam_prob:.1f}% | Ham probability: {ham_prob:.1f}%")

st.divider()
with st.expander("ℹ️ About this model"):
    st.write("""
    - **Dataset**: UCI SMS Spam Collection (5,572 messages)
    - **Technique**: Text cleaning → TF-IDF vectorization → Multinomial Naive Bayes
    - **Test Accuracy**: ~97%
    - You can use this in both your project report and live demo.
    """)