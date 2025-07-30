import streamlit as st
import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import re

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Load CSV from GitHub
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Amhatre-15/SECT-IP/main/data/sentiment_dataset.csv"
    df = pd.read_csv(url, encoding='ISO-8859-1', header=None,
                     names=['target', 'ids', 'date', 'flag', 'user', 'text'])
    return df

# Preprocess the text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)           # remove URLs
    text = re.sub(r"@\w+", "", text)              # remove mentions
    text = re.sub(r"#\w+", "", text)              # remove hashtags
    text = re.sub(r"[^A-Za-z\s]", "", text)       # remove punctuation
    text = text.lower()                           # to lowercase
    text = " ".join([word for word in text.split() if word not in stop_words])  # remove stopwords
    return text

def main():
    st.title("Twitter Sentiment Analysis (Offline CSV)")
    st.write("Choose Option")

    option = st.radio("Select an action", ["Search Tweets by Topic"])
    
    df = load_data()
    df['cleaned_text'] = df['text'].astype(str).apply(clean_text)

    if option == "Search Tweets by Topic":
        query = st.text_input("Enter topic/keyword (e.g., Modi, Olympics, AI):")

        if query:
            matched = df[df['cleaned_text'].str.contains(query, case=False, na=False)]

            if matched.empty:
                st.warning("No tweets matched your query.")
                return

            st.subheader(f"Showing results for '{query}'")
            st.dataframe(matched[['text']].head(10))

            # Basic sentiment prediction
            model = make_pipeline(CountVectorizer(), MultinomialNB())
            model.fit(df['cleaned_text'], df['target'])

            predictions = model.predict(matched['cleaned_text'])
            matched['predicted_sentiment'] = predictions

            sentiment_map = {0: 'Negative', 4: 'Positive'}
            matched['predicted_sentiment'] = matched['predicted_sentiment'].map(sentiment_map)

            st.subheader("Sentiment Distribution")
            fig, ax = plt.subplots()
            sns.countplot(data=matched, x='predicted_sentiment', ax=ax)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
