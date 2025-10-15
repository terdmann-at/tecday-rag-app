"""Qdrant client for embedding storage and retrieval - lots of useful functions to interact with our database"""

import os
import uuid

from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Batch, Distance, VectorParams
from qdrant_client.http.models.models import ScoredPoint
from qdrant_client.qdrant_client import QdrantClient


def instantiate_qclient(
    qdrant_path: str = "",
    qdrant_url: str = "",
    qdrant_port: int = 6333,
    qdrant_api_key: str = "",
) -> QdrantClient:
    """Create db when there's none or instantiate an existing one
    QdrantClient is a client for interacting with Qdrant API

    Args:
        qdrant_url (str): e.g. "http://localhost"
        qdrant_port (int): e.g. 6333
    """

    if qdrant_path:
        return QdrantClient(path=qdrant_path)

    if not qdrant_api_key:
        from dotenv import load_dotenv

        load_dotenv

        qdrant_api_key = os.getenv("QDRANT_API_KEY")

    qclient = QdrantClient(url=qdrant_url, port=qdrant_port, api_key=qdrant_api_key)
    return qclient


def create_db_collection(q_client: QdrantClient, collection_name: str, vectorsize: int):
    """Create db collection within exisiting db

    Args:
        q_client (QdrantClient): qdrant instance
        collection_name (str): preferred collection name
        vectorsize (int): size of embeddings based on used model
    """
    # Create db collection within a given db
    q_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vectorsize, distance=Distance.COSINE),
    )


def delete_db_collection(q_client: QdrantClient, collection_name: str):
    """Delete existing db collection if no longer needed

    Args:
        q_client (QdrantClient): qdrant instance
        collection_name (str): name of collection within db
    """
    # Delete existing db collection
    q_client.delete_collection(collection_name)


def upsert_embeddings(
    q_client: QdrantClient,
    collection_name: str,
    payloads: list[dict],
    embeddings: list[list],
):
    """Upload embeddings to db

    Args:
        q_client (QdrantClient): qdrant instance
        collection_name (str): name of collection within db
        texts (List[str]): payloads to be uploaded
        embeddings (List[List]): text embeddings

    Returns:
        bool: True for successful operation
    """
    # Text ids
    ids = [str(uuid.uuid4()) for _ in range(len(payloads))]

    # Uploading embeddings to database collection
    q_client.upsert(
        collection_name=collection_name,
        points=Batch(ids=ids, payloads=payloads, vectors=embeddings),
    )


def search_documents(
    q_client: QdrantClient, collection_name: str, embedded_question: list[float]
) -> list[ScoredPoint]:
    """Given a question, return X documents ranked on cosine similarity score

    Args:
        q_client (QdrantClient): qdrant instance
        collection_name (str): where embeddings are stored
        embedded_question (List[float]): current question in embedded format

    Returns:
        List[ScoredPoint]: List of payloads per document
    """
    # Searching for the most similar text
    search_result = q_client.search(
        collection_name=collection_name,
        query_vector=embedded_question,
        query_filter=None,
        limit=10,
        with_payload=True,
    )

    return search_result