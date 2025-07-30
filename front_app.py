import streamlit as st
import requests
import joblib
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load the trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Bearer Token - Use Streamlit secrets in production
BEARER_TOKEN = os.getenv("BEARER_TOKEN") or "YOUR_BEARER_TOKEN_HERE"

# Text cleaning and preprocessing
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Predict sentiment
def predict_sentiment(text, model, vectorizer, stop_words):
    clean_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([clean_text])
    prediction = model.predict(vectorized_text)[0]
    return prediction

# Create a colored card for sentiment
def create_card(tweet_text, sentiment):
    color = {
        "positive": "#d4edda",
        "negative": "#f8d7da",
        "neutral": "#fff3cd"
    }.get(sentiment, "#e2e3e5")

    return f"""
    <div style='background-color: {color}; padding: 10px; margin: 10px 0; border-radius: 10px;'>
        <strong>{sentiment.upper()}</strong><br>{tweet_text}
    </div>
    """

# Fetch tweets by topic
def fetch_topic_tweets(topic, max_results=5):
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    params = {
        "query": f"{topic} -is:retweet lang:en",
        "tweet.fields": "created_at",
        "max_results": max_results
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        st.error(f"Error fetching tweets: {response.status_code}")
        return []

# Main app
st.title("Twitter Sentiment Analysis")
option = st.selectbox("Choose an option", ["Input text", "Get tweets on topic"])

if option == "Input text":
    user_input = st.text_area("Enter your text here:")
    if st.button("Analyze Sentiment"):
        sentiment = predict_sentiment(user_input, model, vectorizer, stop_words)
        st.markdown(create_card(user_input, sentiment), unsafe_allow_html=True)

elif option == "Get tweets on topic":
    topic = st.text_input("Enter a topic or keyword (e.g. 'India Budget', 'Elon Musk')")
    if st.button("Fetch Tweets"):
        if topic.strip():
            tweets = fetch_topic_tweets(topic)
            if tweets:
                for tweet in tweets:
                    tweet_text = tweet["text"]
                    sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
                    card_html = create_card(tweet_text, sentiment)
                    st.markdown(card_html, unsafe_allow_html=True)
            else:
                st.info("No tweets found or an error occurred.")
        else:
            st.warning("Please enter a topic or keyword.")
