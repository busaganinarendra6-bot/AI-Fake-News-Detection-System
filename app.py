import streamlit as st
import pickle
import re

# Page Settings
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Load Model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Text Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# Header
st.title("📰 AI-Powered Fake News Detector")
st.markdown("### Detect whether a news article is Real or Fake using Machine Learning")

st.markdown("---")

# Input Box
news = st.text_area(
    "Paste News Article Here",
    height=200,
    placeholder="Enter news article text..."
)

# Prediction
if st.button("🔍 Analyze News"):

    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:

        cleaned_news = clean_text(news)

        vectorized_news = vectorizer.transform([cleaned_news])

        prediction = model.predict(vectorized_news)

        probability = model.predict_proba(vectorized_news)
        confidence = max(probability[0]) * 100

        st.markdown("---")

        if prediction[0] == 0:
            st.error("❌ Prediction: Fake News")
        else:
            st.success("✅ Prediction: Real News")

        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

        st.progress(int(confidence))

st.markdown("---")
st.caption("Built using Python, Scikit-Learn, NLP, TF-IDF and Streamlit")