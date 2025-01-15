import streamlit as st
import matplotlib.pyplot as plt
from utils import load_model, preprocess_text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

# Title and description
st.title("Logistic Regression Sentiment Analysis")
st.write("Analyze the sentiment of user input text using a Logistic Regression model.")

# Text input
user_input = st.text_area("Enter text to analyze:", height=150)

# Load the pre-trained Logistic Regression model
model, vectorizer = load_model('../models/logistic_regression_model.pkl')

# Sentiment analysis using VADER
analyzer = SentimentIntensityAnalyzer()
vader_sentiment = analyzer.polarity_scores(user_input)


if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Preprocess user input
        processed_input = preprocess_text(user_input)
        
        # Vectorize the input text
        input_vectorized = vectorizer.transform([processed_input])
        
        # Extract VADER sentiment scores and text length
        vader_features = [
            vader_sentiment['compound'],  # Compound score
            len(user_input.split())  # Text length (in words)
        ]
        
        # Combine the vectorized input with the additional features
        input_combined = np.hstack([input_vectorized.toarray(), np.array(vader_features).reshape(1, -1)])

        # Predict sentiment and probabilities
        prediction = model.predict(input_combined)[0]
        probabilities = model.predict_proba(input_combined)[0]

        st.subheader("VADER Sentiment Analysis")
        st.write(f"**Positive:** {vader_sentiment['pos']:.4f}")
        st.write(f"**Negative:** {vader_sentiment['neg']:.4f}")
        st.write(f"**Neutral:** {vader_sentiment['neu']:.4f}")
        st.write(f"**Compound:** {vader_sentiment['compound']:.4f}")

        # Display results
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.subheader("Sentiment Result")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Probability:** Positive: {probabilities[1]:.4f}, Negative: {probabilities[0]:.4f}")

        # Visualize probabilities
        st.subheader("Probability Distribution by Logistic Regression")
        fig, ax = plt.subplots()
        ax.bar(["Negative", "Positive"], probabilities, color=["red", "green"])
        ax.set_ylabel("Probability")
        ax.set_title("Sentiment Probability Distribution")
        st.pyplot(fig)
    else:
        st.warning("Please enter some text for analysis.")
