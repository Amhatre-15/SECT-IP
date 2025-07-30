import streamlit as st
import pickle
import re
import requests
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# ====== Load Bearer Token Securely ======
BEARER_TOKEN = st.secrets.get("BEARER_TOKEN", None)

if not BEARER_TOKEN:
    st.error("Twitter Bearer Token is missing. Add it to Streamlit Secrets.")
    st.stop()

# ====== Load Stopwords ======
@st.cache_resource
def load_stopwords():
    return set([
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
        'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
        'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
        'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
        'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
        'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to',
        'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'when',
        'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
        'just', 'don', 'should', 'now'
    ])

# ====== Load Model & Vectorizer ======
@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# ====== Predict Sentiment ======
def predict_sentiment(text, model, vectorizer, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text_vector = vectorizer.transform([text])
    sentiment = model.predict(text_vector)
    return "Negative" if sentiment == 0 else "Positive"

# ====== Fetch Tweets using Twitter API ======
def fetch_user_tweets(username, max_results=5):
    url = f"https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    params = {
        "query": f"from:{username} -is:retweet",
        "tweet.fields": "created_at",
        "max_results": max_results
    }
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        st.error(f"Error fetching tweets: {response.status_code}")
        return []

# ====== Display Colored Card ======
def create_card(tweet_text, sentiment):
    color = "green" if sentiment == "Positive" else "red"
    card_html = f"""
    <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h5 style="color: white;">{sentiment} Sentiment</h5>
        <p style="color: white;">{tweet_text}</p>
    </div>
    """
    return card_html

# ====== Streamlit App ======
def main():
    st.title("Twitter Sentiment Analysis")

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()

    option = st.selectbox("Choose an option", ["Input text", "Get tweets from user"])
    
    if option == "Input text":
        text_input = st.text_area("Enter text to analyze sentiment")
        if st.button("Analyze"):
            if text_input.strip():
                sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
                st.success(f"Sentiment: {sentiment}")
            else:
                st.warning("Please enter some text.")

    elif option == "Get tweets from user":
        username = st.text_input("Enter Twitter username (without @)")
        if st.button("Fetch Tweets"):
            if username.strip():
                tweets = fetch_user_tweets(username)
                if tweets:
                    for tweet in tweets:
                        tweet_text = tweet["text"]
                        sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
                        card_html = create_card(tweet_text, sentiment)
                        st.markdown(card_html, unsafe_allow_html=True)
                else:
                    st.info("No tweets found or user is private.")
            else:
                st.warning("Please enter a Twitter username.")

if __name__ == "__main__":
    main()
