import streamlit as st
import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

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

def preprocess(text, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

def predict_sentiment(texts, model, vectorizer, stop_words):
    processed = [preprocess(t, stop_words) for t in texts]
    features = vectorizer.transform(processed)
    predictions = model.predict(features)
    return ["Negative" if p == 0 else "Positive" for p in predictions]

def create_card(tweet_text, sentiment):
    color = "green" if sentiment == "Positive" else "red"
    card_html = f"""
    <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h5 style="color: white;">{sentiment} Sentiment</h5>
        <p style="color: white;">{tweet_text}</p>
    </div>
    """
    return card_html

def plot_graphs(df):
    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(x='Sentiment', y='Count', data=sentiment_counts, hue='Sentiment',
                palette='pastel', legend=False, ax=axes[0])
    axes[0].set_title('Sentiment Distribution')
    axes[0].set_xlabel('Sentiment')
    axes[0].set_ylabel('Number of Tweets')

    axes[1].pie(sentiment_counts['Count'], labels=sentiment_counts['Sentiment'],
                autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    axes[1].set_title('Sentiment Breakdown (%)')

    st.pyplot(fig)

def main():
    st.title("📊 Twitter Sentiment Analysis (Offline CSV)")

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()

    uploaded_file = st.file_uploader("📁 Upload a CSV file with tweets", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'text' not in df.columns:
            st.error("CSV must have a 'text' column.")
            return
        df = df[['text']]
        df.columns = ['Tweet']

        df['Sentiment'] = predict_sentiment(df['Tweet'], model, vectorizer, stop_words)

        for i in range(min(5, len(df))):
            card_html = create_card(df['Tweet'][i], df['Sentiment'][i])
            st.markdown(card_html, unsafe_allow_html=True)

        plot_graphs(df)

if __name__ == "__main__":
    main()
