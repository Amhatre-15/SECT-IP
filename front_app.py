import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

@st.cache_data
def load_dataset():
    df = pd.read_csv('training.1600000.processed.noemoticon.csv', header=None,
                     names=['sentiment','id','date','query','user','text'], encoding='latin-1')
    df = df[['text', 'sentiment']]
    df['Sentiment'] = df['sentiment'].apply(lambda val: "Positive" if val == 4 else "Negative")
    df.rename(columns={'text': 'Tweet'}, inplace=True)
    df = df.sample(500, random_state=42).reset_index(drop=True)  # Sample only 500 for speed
    return df

@st.cache_resource
def get_analyzer():
    return SentimentIntensityAnalyzer()

def classify_vader(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def plot_sentiment_distribution(df):
    sentiment_counts = df['VADER Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        sns.barplot(x='Sentiment', y='Count', data=sentiment_counts, hue='Sentiment',
                    palette='pastel', legend=False, ax=ax1)
        ax1.set_title('VADER Sentiment Bar Chart')
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.pie(sentiment_counts['Count'], labels=sentiment_counts['Sentiment'],
                autopct='%1.1f%%', colors=sns.color_palette('pastel'))
        ax2.set_title('VADER Sentiment Pie Chart')
        st.pyplot(fig2)

def main():
    st.title("📊 Offline Twitter Sentiment Analysis (VADER + CSV)")

    df = load_dataset()
    analyzer = get_analyzer()

    df['VADER Score'] = df['Tweet'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['Tweet Length'] = df['Tweet'].apply(len)
    df['VADER Sentiment'] = df['VADER Score'].apply(classify_vader)

    st.subheader("Sample Data")
    st.dataframe(df[['Tweet', 'VADER Score', 'VADER Sentiment']].head(10))

    st.subheader("Sentiment Distribution")
    plot_sentiment_distribution(df)

    st.subheader("Tweet Length vs Sentiment")
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    sns.boxplot(x='VADER Sentiment', y='Tweet Length', data=df, palette='pastel')
    ax3.set_title('Tweet Length by Sentiment')
    st.pyplot(fig3)

if __name__ == "__main__":
    main()
