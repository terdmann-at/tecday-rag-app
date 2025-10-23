import glob
import os
from pathlib import Path

import streamlit as st

from utils import get_response_generator, get_retriever

# Check for API key
if not os.getenv("AZURE_OPENAI_API_KEY"):
    st.error(
        "AzureOpenAI API key not found. Please create a `.env` file with `AZURE_OPENAI_API_KEY='...'`"
    )
    st.stop()

pdf_files = list(map(Path, glob.glob("data/*.pdf")))

st.set_page_config(page_title="Chat with Your Document", page_icon="🤖", layout="wide")
st.title("RAG Chatbot")
st.markdown(f"""
Welcome! This chatbot uses a PDF document as its knowledge base.
Ask a question about the content of the knowledge base.

Files currenty loaded:
{"\n* ".join(["", *[f.name if f else "Nothing uploaded." for f in pdf_files]])}
""")

# Get the response generator
try:
    retriever = get_retriever(pdf_files)
    get_response = get_response_generator()

except Exception as e:
    st.error(f"Failed to initialize the RAG system: {e}")
    st.info(
        "Please ensure your OpenAI API key is correct and there are PDF files in `/data`."
    )
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = get_response(prompt, retriever, messages=st.session_state.messages)
                answer = response.answer
                sources = response.sources
                st.markdown(answer)

                if sources:
                    st.markdown("---")
                    st.markdown("### Sources")

                    for i, source in enumerate(sources):
                        source_label = f"{source.file} (page: {source.page})"
                        with st.expander(source_label):
                            st.code(source.text, language="text")

                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
