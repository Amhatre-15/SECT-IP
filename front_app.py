import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from ntscraper import Nitter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 📦 Load NLTK stopwords
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# 🔍 Load vectorizer + model
@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# 🧠 Predict sentiment
def predict_sentiment(text, model, vectorizer, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text = [text]
    text = vectorizer.transform(text)
    sentiment = model.predict(text)
    return "Negative" if sentiment == 0 else "Positive"

# 💬 UI tweet card
def create_card(tweet_text, sentiment):
    color = "green" if sentiment == "Positive" else "red"
    return f"""
    <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h5 style="color: white;">{sentiment} Sentiment</h5>
        <p style="color: white;">{tweet_text}</p>
    </div>
    """

# 📊 Graphs (bar + pie)
def plot_sentiment_charts(df):
    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    st.subheader("📊 Sentiment Distribution")

    fig1, ax1 = plt.subplots()
    sns.barplot(x='Sentiment', y='Count', data=sentiment_counts, hue='Sentiment',
                palette='pastel', legend=False, ax=ax1)
    ax1.set_title('Sentiment Distribution')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.pie(sentiment_counts['Count'], labels=sentiment_counts['Sentiment'],
            autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    ax2.set_title('Sentiment Breakdown (%)')
    st.pyplot(fig2)

# 🚀 Main app logic
def main():
    st.title("Twitter Sentiment Analysis 🧠")

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    scraper = Nitter(log_level=1)

    option = st.selectbox("Choose an option", ["Input text", "Get tweets from user"])

    if option == "Input text":
        text_input = st.text_area("Enter text to analyze sentiment")
        if st.button("Analyze"):
            sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
            st.success(f"Sentiment: {sentiment}")
            df = pd.DataFrame({'Sentiment': [sentiment]})
            plot_sentiment_charts(df)

    elif option == "Get tweets from user":
        username = st.text_input("Enter Twitter username")
        if st.button("Fetch Tweets"):
            tweets_data = scraper.get_tweets(username, mode='user', number=10)
            if 'tweets' in tweets_data:
                sentiments = []
                for tweet in tweets_data['tweets']:
                    tweet_text = tweet['text']
                    sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
                    sentiments.append({'Tweet': tweet_text, 'Sentiment': sentiment})
                    st.markdown(create_card(tweet_text, sentiment), unsafe_allow_html=True)
                df = pd.DataFrame(sentiments)
                plot_sentiment_charts(df)
            else:
                st.error("No tweets found or an error occurred.")

if __name__ == "__main__":
    main()
