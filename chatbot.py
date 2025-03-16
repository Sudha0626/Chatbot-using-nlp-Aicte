import os
import json
import datetime
import csv
import random
import time
import nltk
import ssl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fix SSL Issue for NLTK
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Load intents from JSON
file_path = os.path.abspath("C:/Users/HOME/Desktop/aicte_chatbot_nlp/intents.json")
with open(file_path, "r") as file:
    data = json.load(file)
    intents = data["intents"]

# Initialize vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess training data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Function to get chatbot response
def chatbot(input_text):
    input_text_vectorized = vectorizer.transform([input_text])
    probabilities = clf.predict_proba(input_text_vectorized)
    predicted_tag = clf.predict(input_text_vectorized)[0]
    
    # Debugging: Print the confidence scores
    print("\nDEBUG: Predicted Tag =", predicted_tag)
    print("DEBUG: Confidence Scores =", probabilities)

    for intent in intents:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Custom CSS for better UI
st.markdown("""
    <style>
    .user-bubble {
        background-color: #0078FF;
        color: white;
        padding: 10px;
        border-radius: 15px;
        max-width: 60%;
        text-align: right;
        margin-left: auto;
        margin-bottom: 10px;
        display: block;
    }
    .bot-bubble {
        background-color: #E0E0E0;
        color: black;
        padding: 10px;
        border-radius: 15px;
        max-width: 60%;
        text-align: left;
        margin-right: auto;
        margin-bottom: 10px;
        display: block;
    }
    .chat-container {
        height: 400px;
        overflow-y: auto;
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 10px;
        background-color: #f9f9f9;
        margin-bottom: 20px;
    }
    .input-container {
        margin-top: 15px;
        padding-top: 10px;
        border-top: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Menu
menu = ["Home", "Conversation History", "About"]
choice = st.sidebar.selectbox("Menu", menu)

# Function to handle message sending
def send_message():
    if st.session_state.user_input.strip() != "":
        # Store user message
        st.session_state.messages.append({"role": "user", "content": st.session_state.user_input})
        
        # Simulate chatbot thinking delay
        time.sleep(0.5)
        response = chatbot(st.session_state.user_input)
        
        # Store chatbot response
        st.session_state.messages.append({"role": "bot", "content": response})

        # Save conversation to CSV
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("chat_log.csv", "a", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([st.session_state.user_input, response, timestamp])

        # **CLEAR INPUT FIELD AFTER SENDING**
        st.session_state.user_input = ""

# Home Page
if choice == "Home":
    st.title("🤖 AI Chatbot")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

    # Typing box with "Send" button
    st.text_input("Type your message:", key="user_input", on_change=send_message)

# Conversation History Page
elif choice == "Conversation History":
    st.title("📜 Conversation History")

    if os.path.exists("chat_log.csv"):
        with open("chat_log.csv", "r", encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")
    else:
        st.write("No conversation history found.")

# About Page
elif choice == "About":
    st.title("ℹ️ About the Chatbot")
    st.write("""
    This AI-powered chatbot is built using Natural Language Processing (NLP) and Machine Learning.
    - Uses **Logistic Regression** for intent classification.
    - **Streamlit** is used for the interactive chatbot UI.
    - Stores **chat history** for later review.
    
    💡 This chatbot can be extended with deep learning techniques for better accuracy.
    """)

