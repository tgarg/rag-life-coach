import streamlit as st
from src.ollama_client import OllamaClient
from src.chat_interface import ChatInterface

# Initialize session state. If the LLM isn't running, it will be started here. If there isn't a chat interface object, it will be created here.
if "chat" not in st.session_state:
    st.session_state.chat = ChatInterface()
if "ollama" not in st.session_state:
    st.session_state.ollama = OllamaClient()

st.title("Mistral Chatbot (Ollama + Streamlit)")

# Each message in the chat history is displayed in the chat interface. The role (user or assistant) determines the alignment of the message.
for msg in st.session_state.chat.get_history():
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("Say something..."):
    st.session_state.chat.add_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.ollama.generate(prompt)
            st.markdown(response)
            st.session_state.chat.add_message("assistant", response)