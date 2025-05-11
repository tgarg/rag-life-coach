import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
import streamlit as st
import json
from src.ollama_client import OllamaClient
from src.chat_interface import ChatInterface
from src.retriever import get_relevant_chunks
from langchain_huggingface import HuggingFaceEmbeddings
from src.profile_generator import generate_json_profile, load_all_journal_entries

# Function to load or generate the user's core profile
def load_or_generate_user_profile(profile_path="user_profile.json"):
    try:
        # Try to load existing profile
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)
            # Convert the JSON object to a string for easy inclusion in the prompt
            # You can also choose to format this string differently or select parts of it
            return json.dumps(profile_data, indent=2, ensure_ascii=False)
    except FileNotFoundError:
        # If profile doesn't exist, generate it
        st.info("User profile not found. Generating profile from journal entries...")
        journal_data = load_all_journal_entries("journals")
        if journal_data:
            profile = generate_json_profile(journal_data)
            if profile:
                try:
                    with open(profile_path, "w", encoding="utf-8") as f:
                        json.dump(profile, f, indent=2, ensure_ascii=False)
                    st.success("Successfully generated user profile!")
                    return json.dumps(profile, indent=2, ensure_ascii=False)
                except Exception as e:
                    st.error(f"Error saving generated profile: {e}")
            else:
                st.error("Failed to generate user profile.")
        else:
            st.error("No journal entries found to generate profile.")
        return "User profile generation failed. Please check your journal entries and try again."
    except json.JSONDecodeError:
        return "Error parsing user_profile.json. Ensure it's a valid JSON."
    except Exception as e:
        return f"Error loading user profile: {e}"

# Initialize session state. If the LLM isn't running, it will be started here. If there isn't a chat interface object, it will be created here.
if "chat" not in st.session_state:
    st.session_state.chat = ChatInterface()
if "ollama" not in st.session_state:
    st.session_state.ollama = OllamaClient()

# Load or generate user core info once per session
if "user_core_info" not in st.session_state:
    st.session_state.user_core_info = load_or_generate_user_profile()

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

    context = "\n\n".join(get_relevant_chunks(prompt))
    user_core_info_content = st.session_state.user_core_info

    # --- Prompt with RAG and memory ---
    rag_prompt = f"""
    You are a thoughtful, compassionate life coach who helps the user reflect on their thoughts, patterns, and goals.

    Here is a JSON representation of the user's core profile, derived from their journal entries. This information provides standing context about their values, patterns, goals, and challenges. It is always relevant:
    --- USER CORE PROFILE (JSON) ---
    {user_core_info_content}
    --- END USER CORE PROFILE (JSON) ---

    Here are some of the user's past journal entries that may be relevant to the current conversation:
    --- RELEVANT JOURNAL EXCERPTS ---
    {context}
    --- END RELEVANT JOURNAL EXCERPTS ---

    Now, the user asks:
    "{prompt}"

    Based on the USER CORE PROFILE (JSON), the RELEVANT JOURNAL EXCERPTS, and your understanding of the user, respond with insight, follow-up questions, or gentle encouragement. Ensure you consider the USER CORE PROFILE (JSON) in every response.
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

