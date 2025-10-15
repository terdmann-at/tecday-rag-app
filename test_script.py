import streamlit as st

from rag_llm_applications.chat import get_chat_generator
from rag_llm_applications.embed import get_embedder
from rag_llm_applications.model import Document

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from rag_llm_applications.qdrant import create_db_collection, instantiate_qclient, delete_db_collection
from rag_llm_applications.util import PROJECT_ROOT, load_config, logging_setup

load_dotenv()


config = load_config(PROJECT_ROOT / "config" / "config.yml")

embedder = config["embedder"]
embedder = get_embedder(embedder)

chat_generator = config["chat_generator"]
chat_generator = get_chat_generator(chat_generator)


print("This looks good.")