import streamlit as st
import pandas as pd
import pickle
import re

# Load model and vectorizer
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Load pre-sampled tweet dataset
@st.cache_data
def load_data():
    df = pd.read_csv("sample_data.csv")
    return df

df = load_data()

# Clean text for vectorization
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = text.lower()
    return text

# Streamlit App
st.set_page_config(page_title="Twitter Sentiment Search", layout="centered")
st.title("🔍 Twitter Sentiment Finder")
st.markdown("Enter a topic/word to search tweets and analyze their sentiment.")

# Input
query = st.text_input("Enter keyword to search:")

if query:
    # Filter tweets containing the query
    matched = df[df['text'].str.contains(query, case=False, na=False)]
    
    if not matched.empty:
        st.write(f"Found {len(matched)} matching tweets:")
        
        # Predict sentiment
        matched['clean'] = matched['text'].apply(clean_text)
        vec = vectorizer.transform(matched['clean'])
        matched['pred'] = model.predict(vec)
        matched['sentiment'] = matched['pred'].apply(lambda x: "😊 Positive" if x == 4 else "☹️ Negative")

        # Show results
        for i, row in matched.head(10).iterrows():  # Show first 10 results
            st.write(f"**Tweet:** {row['text']}")
            st.write(f"**Sentiment:** {row['sentiment']}")
            st.markdown("---")
    else:
        st.warning("No matching tweets found.")
