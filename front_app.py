import streamlit as st
import pandas as pd
import gdown
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download NLTK data
nltk.download('vader_lexicon')

# Cache the data loading process
@st.cache_data
def download_and_load_csv():
    url = "https://drive.google.com/uc?id=1Xe7bGaxm9qxMxHiUMh03QhDwIWyBNIEg"
    output = "sentiment_data.csv"
    gdown.download(url, output, quiet=False)

    # Use correct encoding and assign proper column names
    column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(output, encoding='ISO-8859-1', header=None, names=column_names)
    return df

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def main():
    st.title("Twitter Sentiment Analysis")
    st.write("### Choose Option")
    st.write("**Search Tweets by Topic**")

    # Get input
    query = st.text_input("Enter topic/keyword (e.g., Modi, Olympics, AI):")

    if query:
        df = download_and_load_csv()

        # Show available columns for debugging
        st.write("✅ Columns in dataset:", df.columns.tolist())

        # Filter relevant tweets
        matched = df[df['text'].str.contains(query, case=False, na=False)]

        if matched.empty:
            st.warning("❌ No matching tweets found for the topic.")
            return

        # Apply sentiment analysis
        matched['Sentiment'] = matched['text'].apply(analyze_sentiment)

        # Display results
        st.write(f"### Showing {len(matched)} matching tweets for: `{query}`")
        st.dataframe(matched[['text', 'Sentiment']].head(20))

        # Plot sentiment distribution
        sentiment_counts = matched['Sentiment'].value_counts()
        st.write("### Sentiment Distribution")
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', ax=ax, color=['green', 'red', 'gray'])
        st.pyplot(fig)

if __name__ == "__main__":
    main()
