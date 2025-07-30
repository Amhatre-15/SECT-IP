import streamlit as st
import pickle
import re

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the vectorizer (Tfidf or CountVectorizer)
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Function to clean the user input
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # remove URLs
    text = re.sub(r'\@w+|\#','', text)  # remove mentions and hashtags
    text = re.sub(r"[^a-zA-Z\s]", '', text)  # remove special characters
    text = text.lower()
    return text

# Streamlit app layout
st.set_page_config(page_title="Twitter Sentiment Analysis", layout="centered")
st.title("📊 Twitter Sentiment Analysis")
st.markdown("Enter a tweet or sentence below to analyze its **sentiment** based on a pre-trained model.")

# Input box
user_input = st.text_input("🔍 Enter text here:")

# Prediction
if user_input:
    cleaned = clean_text(user_input)
    vectorized_input = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized_input)[0]

    if prediction == 0:
        st.error("☹️ Negative Sentiment")
    elif prediction == 4:
        st.success("😊 Positive Sentiment")
    else:
        st.warning("😐 Neutral/Unknown Sentiment")
