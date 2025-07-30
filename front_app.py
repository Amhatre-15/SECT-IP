import streamlit as st
import requests
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

# Preprocessing + Prediction
def predict_sentiment(text, model, vectorizer, stop_words):
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)
    return "Positive" if prediction == 1 else "Negative"

# Twitter API Request
def fetch_tweets(query, bearer_token, count=5):
    url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results={count}&tweet.fields=text"
    headers = {
        "Authorization": f"Bearer {bearer_token}"
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return [tweet['text'] for tweet in response.json().get("data", [])]
    else:
        st.error(f"Error fetching tweets: {response.status_code}")
        return []

# Streamlit UI
def main():
    st.title("Twitter Sentiment Analysis")

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()

    option = st.selectbox("Choose Option", ["Input Text", "Search Tweets by Topic"])
    
    if option == "Input Text":
        text = st.text_area("Enter your text:")
        if st.button("Analyze"):
            sentiment = predict_sentiment(text, model, vectorizer, stop_words)
            st.success(f"Sentiment: {sentiment}")

    else:
        query = st.text_input("Enter topic/keyword (e.g., Modi, Olympics, AI):")
        bearer_token = st.text_input("Enter your Twitter API Bearer Token", type="password")
        if st.button("Fetch Tweets"):
            tweets = fetch_tweets(query, bearer_token)
            if tweets:
                for tweet in tweets:
                    sentiment = predict_sentiment(tweet, model, vectorizer, stop_words)
                    st.markdown(f"**{sentiment}**: {tweet}")
            else:
                st.warning("No tweets found or an error occurred.")

if __name__ == "__main__":
    main()
