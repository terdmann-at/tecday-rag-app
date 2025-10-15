import streamlit as st
from rag_llm_applications.rag import RAG


st.title("My RAG Chatbot")


# TASK: Do the exercises and use the rag class here.
def LLM_response(user_input):
    rag = RAG("data/qdrant", "documents")
    response = rag.generate_answer(user_input).answer
    rag.qdrant.close()
    return response.content


user_input = st.text_area("What do you want to know?", "")

if st.button("Generate"):
    st.write("LLM Response:")

    response = LLM_response(user_input)

    st.write(response)
    st.write("Retrieved text:")
else:
    st.warning("Please enter some text.")
