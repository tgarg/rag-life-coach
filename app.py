import streamlit as st
from src.ollama_client import OllamaClient
from src.chat_interface import ChatInterface
from src.retriever import get_relevant_chunks


# Initialize session state. If the LLM isn't running, it will be started here. If there isn't a chat interface object, it will be created here.
if "chat" not in st.session_state:
    st.session_state.chat = ChatInterface()
if "ollama" not in st.session_state:
    st.session_state.ollama = OllamaClient()

st.title("Journal-Aware Coach")

# Each message in the chat history is displayed in the chat interface. The role (user or assistant) determines the alignment of the message.
for msg in st.session_state.chat.get_history():
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
# Accept user input
if prompt := st.chat_input("Say something..."):
    # Store user message
    st.session_state.chat.add_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

  # --- RAG version ---
    context = "\n\n".join(get_relevant_chunks(prompt))
    rag_prompt = f"""
    You are a thoughtful, compassionate life coach who helps the user reflect on their thoughts, patterns, and goals.

    Here are some of the user's past journal entries that may be relevant:
    ---
    {context}
    ---

    Now, the user asks:
    "{prompt}"

    Based on the journals and your understanding of the user, respond with insight, follow-up questions, or gentle encouragement.
    """

    with st.chat_message("assistant"):
        with st.spinner("Thinking (with memory)..."):
            rag_response = st.session_state.ollama.generate(rag_prompt)
            st.markdown(f"**With memory:**\n\n{rag_response}")

    # # --- Non-RAG version ---
    # non_rag_prompt = f"You are a thoughtful, compassionate life coach. The user asks: \"{prompt}\". Respond with insight, follow-up questions, or gentle encouragement."

    # with st.chat_message("assistant"):
    #     with st.spinner("Thinking (without memory)..."):
    #         non_rag_response = st.session_state.ollama.generate(non_rag_prompt)
    #         st.markdown(f"**Without memory:**\n\n{non_rag_response}")

    # Store both assistant responses
    st.session_state.chat.add_message("assistant", f"**With memory:**\n\n{rag_response}")
    # st.session_state.chat.add_message("assistant", f"**Without memory:**\n\n{non_rag_response}")

