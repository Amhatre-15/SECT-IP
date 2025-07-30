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
    return pd.read_csv(output, encoding="ISO-8859-1")  # handles Unicode error

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

        st.write("Available columns in dataset:", df.columns.tolist())  # DEBUG: show column names

        # Try to detect the correct column
        possible_text_cols = [col for col in df.columns if 'text' in col.lower() or 'tweet' in col.lower()]
        if possible_text_cols:
            col_name = possible_text_cols[0]
            st.success(f"Using column: {col_name} for tweet analysis")

            query = st.text_input("Enter keyword to search tweets (e.g., India, tech, movie):")
            max_results = st.slider("Number of tweets to analyze", 1, 50, 10)

            if st.button("Search and Analyze"):
                matched = df[df[col_name].astype(str).str.contains(query, case=False, na=False)]
                if not matched.empty:
                    for i, row in matched.head(max_results).iterrows():
                        sentiment = predict_sentiment(row[col_name], model, vectorizer, stop_words)
                        st.markdown(f"**{sentiment}**: {row[col_name]}")
                else:
                    st.warning("No matching tweets found.")
        else:
            st.error("❌ Could not find a suitable 'text' or 'tweet' column in your dataset.")

if __name__ == "__main__":
    main()
