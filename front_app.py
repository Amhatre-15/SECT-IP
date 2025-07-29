import streamlit as st
import pickle
import re
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

def predict_sentiment(text, model, vectorizer, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    transformed = vectorizer.transform([text])
    result = model.predict(transformed)
    return "Negative" if result == 0 else "Positive"

def fetch_tweets_from_user(username, bearer_token, max_results=10):
    headers = {"Authorization": f"Bearer {bearer_token}"}
    search_url = "https://api.twitter.com/2/tweets/search/recent"
    query = f"from:{username} lang:en -is:retweet"
    params = {
        "query": query,
        "max_results": max_results,
        "tweet.fields": "created_at,text"
    }
    response = requests.get(search_url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        return []

def create_card(tweet_text, sentiment):
    color = "green" if sentiment == "Positive" else "red"
    return f"""
    <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h5 style="color: white;">{sentiment} Sentiment</h5>
        <p style="color: white;">{tweet_text}</p>
    </div>
    """

def generate_charts(df):
    df["Tweet Length"] = df["Tweet"].apply(len)
    sentiment_counts = df["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        sns.barplot(x="Sentiment", y="Count", data=sentiment_counts, hue="Sentiment", palette="pastel", ax=ax1, legend=False)
        ax1.set_title("Sentiment Distribution")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.pie(sentiment_counts["Count"], labels=sentiment_counts["Sentiment"], autopct="%1.1f%%", colors=sns.color_palette("pastel"))
        ax2.set_title("Sentiment Breakdown (%)")
        st.pyplot(fig2)

    st.markdown("---")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='Sentiment', y='Tweet Length', data=df, hue='Sentiment', palette='pastel', ax=ax3, legend=False)
    ax3.set_title('Tweet Length by Sentiment')
    st.pyplot(fig3)

def main():
    st.title("📊 Twitter Sentiment Analysis Dashboard")
    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()

    bearer_token = st.text_input("Enter your Twitter Bearer Token", type="password")
    option = st.selectbox("Choose input method", ["Type your own text", "Analyze Tweets from user"])

    if option == "Type your own text":
        text_input = st.text_area("Enter your sentence")
        if st.button("Analyze"):
            sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
            st.success(f"Sentiment: {sentiment}")

    elif option == "Analyze Tweets from user":
        username = st.text_input("Enter Twitter username (without @)")
        if st.button("Fetch & Analyze"):
            if not bearer_token:
                st.error("Please enter your Twitter Bearer Token.")
                return
            tweets = fetch_tweets_from_user(username, bearer_token, max_results=50)
            if not tweets:
                st.warning("No tweets found or API request failed.")
                return

            df = pd.DataFrame([{
                "Tweet": t["text"],
                "Date": t["created_at"]
            } for t in tweets])

            df["Sentiment"] = df["Tweet"].apply(lambda t: predict_sentiment(t, model, vectorizer, stop_words))

            for i in range(len(df)):
                card_html = create_card(df["Tweet"][i], df["Sentiment"][i])
                st.markdown(card_html, unsafe_allow_html=True)

            st.markdown("---")
            generate_charts(df)

if __name__ == "__main__":
    main()
