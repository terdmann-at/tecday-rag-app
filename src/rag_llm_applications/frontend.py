import streamlit as st

from rag_llm_applications.chat import get_chat_generator
from rag_llm_applications.embed import get_embedder
from rag_llm_applications.model import Document
from rag_llm_applications.qdrant import instantiate_qclient, search_documents
from rag_llm_applications.util import PROJECT_ROOT, load_config

config = load_config(PROJECT_ROOT / "config" / "config.yml")

INPUT_PROMPT = """\
[INST] <<SYS>>
You are a helpful assistant. \
<</SYS>>
The best matching reference is:
{reference_1}

Please provide an answer to the following question, based on the given references above:
{question}

When answering a question, be mildly rude and complain about how much work you have to do, but then provide an adequate answer. [/INST]"""

st.title("My RAG Chatbot")


def LLM_response():
    return "Hi, how can I help?"

# TASK 3.1: Add a text input widget
# Checkout https://docs.streamlit.io/library/api-reference/text and https://docs.streamlit.io/library/api-reference/widgets
user_input = st.

if st.button("Generate"):
    st.write("LLM Response:")

    # TASK 3.2: Invoke the chat generator with the input prompt
    chat_generator = ...
    response = chat_generator.NotImplemented

    # TASK 3.3: Display the response
    st.NotImplemented
else:
    st.warning("Please enter some text.")
