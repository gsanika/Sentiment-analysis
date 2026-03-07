import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Clean text (same as training)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join(text.split())

# UI
st.title("😊 Sentiment Analysis")
st.write("Predicts: Positive, Negative, Neutral, Irrelevant")

user_input = st.text_area("Enter text:")

if st.button("Predict") and user_input.strip():
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    st.success(f"**Sentiment:** {pred}  \n**Confidence:** {prob:.2f}")