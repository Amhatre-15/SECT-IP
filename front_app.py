import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Page title
st.title("🇮🇳 Indian Topics - Twitter Sentiment Analysis")

# Load the dataset
uploaded_file = 'indian_topics_test_data_nolabel.csv'

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    return df

df = load_data(uploaded_file)
st.subheader("📄 Sample Tweets")
st.write(df.head(5))

# VADER Analyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis
df["Sentiment"] = df["Tweet"].apply(get_sentiment)

# Visualization
st.subheader("📊 Sentiment Distribution")

sentiment_counts = df['Sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']

col1, col2 = st.columns(2)

with col1:
    sns.set_style("whitegrid")
    fig1, ax1 = plt.subplots()
    sns.barplot(x='Sentiment', y='Count', data=sentiment_counts, hue='Sentiment', palette='pastel', legend=False)
    ax1.set_title("Sentiment Count")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.pie(sentiment_counts['Count'], labels=sentiment_counts['Sentiment'],
            autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    ax2.set_title("Sentiment Breakdown")
    st.pyplot(fig2)

# Optional: Show full data with sentiments
st.subheader("🧾 All Tweets with Sentiment")
st.dataframe(df[['Tweet', 'Sentiment']])
