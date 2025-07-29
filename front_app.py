import streamlit as st
import pandas as pd
import pickle
import re
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

def preprocess(text, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

def predict_sentiment(text, model, vectorizer):
    transformed = vectorizer.transform([text])
    sentiment = model.predict(transformed)
    return "Negative" if sentiment == 0 else "Positive"

def main():
    st.title("📊 CSV-Based Sentiment Analysis")

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "Tweet" not in df.columns:
                st.error("CSV must contain a column named 'Tweet'")
                return

            df['Cleaned'] = df['Tweet'].apply(lambda x: preprocess(str(x), stop_words))
            df['Sentiment'] = df['Cleaned'].apply(lambda x: predict_sentiment(x, model, vectorizer))

            st.success("Sentiment Analysis Completed")
            st.dataframe(df[['Tweet', 'Sentiment']])

            sentiment_counts = df['Sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']

            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots()
                sns.barplot(x='Sentiment', y='Count', data=sentiment_counts, palette='pastel', ax=ax1)
                ax1.set_title("Sentiment Distribution")
                st.pyplot(fig1)

            with col2:
                fig2, ax2 = plt.subplots()
                ax2.pie(sentiment_counts['Count'], labels=sentiment_counts['Sentiment'], autopct='%1.1f%%', colors=sns.color_palette('pastel'))
                ax2.set_title("Sentiment Breakdown (%)")
                st.pyplot(fig2)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
