"""Embeddings are used to convert text into vectors.
This module provides a wrapper around the OpenAI functions to embed text.
We will use this to embed the user's question and the document documents, by using the `embed_query` and `embed_documents` methods.
As you will see, we will import the get_embedder function from this module in the frontend file."""

import logging
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings

logger = logging.getLogger("rag_llm_applications")

load_dotenv()


def get_embedder(embedder_name: str = "text-embedding-ada-002") -> AzureOpenAIEmbeddings:
    """Obtain an instance of the OpenAIEmbeddings class - a wrapper around the OpenAI API embedding functionality.

    Args:
        embedder_name (str): type of embedder to use

    Raises:
        ValueError: if the embedder_name is not a valid embedder name

    Returns:
        OpenAIEmbeddings: an instance of the OpenAIEmbeddings class
    """
    if embedder_name == "text-embedding-ada-002":
        return OpenAIEmbeddings(
            model=embedder_name,  # just the model name
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        raise ValueError(f"{embedder_name} is not a valid embedder name")
