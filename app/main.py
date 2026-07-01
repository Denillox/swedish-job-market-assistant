import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from assistant.agent import chat

st.title("Swedish AI/Data Job Market Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

for message in st.session_state.history:
    st.chat_message(message["role"]).write(message["content"])

user_input = st.chat_input("Ask about the Swedish AI/Data job market...")

if user_input:
    st.chat_message("user").write(user_input)

    tuple_history = [
        (message["role"], message["content"]) for message in st.session_state.history
    ]

    reply, updated_tuple_history = chat(user_input, tuple_history)

    st.session_state.history = [
        {"role": role, "content": content} for role, content in updated_tuple_history
    ]

    st.chat_message("assistant").write(reply)