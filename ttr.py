import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Function to preprocess and predict new text
def predict_sentiment(text):
    # Load the tokenizer
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # Load max_length
    with open('max_length.pkl', 'rb') as handle:
        max_length = pickle.load(handle)
    
    # Load the model
    model = load_model('sentiment_lstm_model.h5')
    
    # Preprocess the text
    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_seq, maxlen=max_length, padding='post')
    
    # Make prediction
    prediction = model.predict(text_pad)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Map predicted class back to descriptive sentiment labels
    reverse_sentiment_mapping = {0: 'Agnostic', 1: 'Neutral', 2: 'Believer', 3: 'News'}
    predicted_sentiment = reverse_sentiment_mapping[predicted_class]
    
    return predicted_sentiment

# Streamlit interface
st.title('Sentiment Analysis App')

user_input = st.text_area("Enter a message for sentiment prediction:")

if st.button('Predict'):
    sentiment = predict_sentiment(user_input)
    st.write(f"Predicted sentiment for '{user_input}': {sentiment}")
