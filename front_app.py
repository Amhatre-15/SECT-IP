import streamlit as st
import pandas as pd
import gdown
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Setup
@st.cache_resource
def load_model_and_vectorizer():
    with open("model.pkl", "rb") as m, open("vectorizer.pkl", "rb") as v:
        return pickle.load(m), pickle.load(v)

@st.cache_resource
def load_stopwords():
    nltk.download("stopwords")
    return stopwords.words("english")

@st.cache_data
def download_and_load_csv():
    url = "https://drive.google.com/uc?id=1SekoMdcYy8gpcF7Al8IaKGfkVNHSXSun"
    output = "tweets_data.csv"
    gdown.download(url, output, quiet=False)
    return pd.read_csv(output)

# Preprocessing + Prediction
def predict_sentiment(text, model, vectorizer, stop_words):
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)
    return "Positive" if prediction == 1 else "Negative"

# Streamlit UI
def main():
    st.title("Twitter Sentiment Analysis (Offline Dataset)")

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()

    option = st.selectbox("Choose Option", ["Input Text", "Search in Offline Dataset"])

    if option == "Input Text":
        text = st.text_area("Enter your text:")
        if st.button("Analyze"):
            sentiment = predict_sentiment(text, model, vectorizer, stop_words)
            st.success(f"Sentiment: {sentiment}")

    elif option == "Search in Offline Dataset":
        st.info("Fetching dataset from Google Drive (only once)...")
        df = download_and_load_csv()

        query = st.text_input("Enter keyword to search tweets (e.g., India, tech, movie):")
        max_results = st.slider("Number of tweets to analyze", 1, 50, 10)

        if st.button("Search and Analyze"):
            matched = df[df['text'].str.contains(query, case=False, na=False)]
            if not matched.empty:
                for i, row in matched.head(max_results).iterrows():
                    sentiment = predict_sentiment(row['text'], model, vectorizer, stop_words)
                    st.markdown(f"**{sentiment}**: {row['text']}")
            else:
                st.warning("No matching tweets found.")

if __name__ == "__main__":
    main()
