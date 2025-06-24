import streamlit as st

import os
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()  # take environment variables

# from .env file
# Load environment variables from .env file

token = os.getenv("SECRET")  # Replace with your actual token
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

# initialize the OpenAI client
# Note: The OpenAI client is initialized with the base URL and API key.
client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

st.title("Echo Bot")

if "messages" not in st.session_state:
    st.session_state.messages = [
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What is up?")
# React to user input
if prompt:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    messages_to_send = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    messages_to_send_with_history = messages_to_send + st.session_state.messages

    # call to openai api
    response = client.chat.completions.create(
        messages=messages_to_send_with_history,
        temperature=1.0,
        top_p=1.0,
        model=model
    )

    response_text = response.choices[0].message.content # type: ignore
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response_text)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})