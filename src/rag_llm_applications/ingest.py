"""This is a central part of the setup which reads text from the PDFs, embeds them, and uploads them to Qdrant.
Because understanding this is some important, we have some tasks for you to complete.
Some of these tasks require you to modify methods in model.py"""

import logging
from dataclasses import asdict
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter

from rag_llm_applications.embed import get_embedder
from rag_llm_applications.model import Document
from rag_llm_applications.qdrant import QdrantClient, instantiate_qclient, upsert_embeddings
from rag_llm_applications import create_collection
from rag_llm_applications.util import (
    PROJECT_ROOT,
    load_config,
    logging_setup,
    delete_all_files_in_directory,
)

# TASK 2.1: Import a text parser from langchain.document_loaders.pdf
# Check this out: https://python.langchain.com/docs/how_to/document_loader_pdf/
from langchain_community.document_loaders import PyMuPDFLoader

logger = logging.getLogger("rag_llm_applications")


def parse_documents_as_json(
    source_dir: str | Path,
    target_dir: str | Path,
    text_splitter: TextSplitter = RecursiveCharacterTextSplitter(),
):
    """Parse and split documents from source_dir and save them to target_dir as jsons."""

    raw_documents = Path(source_dir).glob("*.pdf")
    for raw_document in raw_documents:
        logger.debug("Parsing pdf at %s", raw_document)
        # TASK 2.2: Use the text parser of your choice that you imported above to read the text from the raw_document PDF
        pdf_loader = PyMuPDFLoader(str(raw_document))
        # TASK 2.3: find a method to load and chunk text in your text parser
        # Check the API here: https://python.langchain.com/api_reference/
        pdf_pages = pdf_loader.load_and_split(text_splitter=text_splitter)
        for page in pdf_pages:
            document = Document.from_langchain_document(page)
            document.save(target_dir)


def update_document_jsons_with_embedding(source_dir: str | Path, target_dir: str | Path, embedder):
    """Embeds the documents and saves them to the target_dir."""
    logger.info("Loading documents from %s", source_dir)
    preprocessed_documents = Path(source_dir).glob("*.json")
    documents_without_embedding = [
        # TASK 2.4: Check the from_json_file method to Document class
        Document.from_json_file(preprocessed_document)
        for preprocessed_document in preprocessed_documents
    ]

    document_texts = [document.content for document in documents_without_embedding]
    # TASK 2.5: Get embeddings for document_texts
    embeddings = embedder.embed_documents(document_texts)
    logger.info("Embedded %s documents", len(embeddings))

    logging.info("Saving documents to %s", target_dir)
    for document, embedding in zip(documents_without_embedding, embeddings):
        document.add_embedding(embedding)
        document.save(target_dir)


def upload_document_jsons_to_qdrant(
    source_dir: str | Path,
    qdrant_collection: str,
    qdrant_client: QdrantClient,
):
    """Uploads embedded documents (vectors) to Qdrant."""
    logger.info("Loading documents from %s", source_dir)
    embedded_documents = Path(source_dir).glob("*.json")
    documents = [
        Document.from_json_file(embedded_document) for embedded_document in embedded_documents
    ]

    logger.info("Uploading %s documents to Qdrant", len(documents))
    document_data_as_dict = [asdict(document) for document in documents]
    embeddings = [document.pop("embedding") for document in document_data_as_dict]

    # TASK 2.6: Batch upsert the embeddings to qdrant
    # Hint: check out the function upsert_embeddings in qdrant.py
    upsert_embeddings(qdrant_client, qdrant_collection, document_data_as_dict, embeddings)


def main(text_splitter: TextSplitter = RecursiveCharacterTextSplitter()):
    """Parse and split documents, embed them, and upload them to Qdrant."""
    config = load_config(PROJECT_ROOT / "config" / "config.yml")
    logging_setup(config)

    # wipe everything clean
    delete_all_files_in_directory(config["paths"]["data"]["preprocessed"])
    delete_all_files_in_directory(config["paths"]["data"]["embedded"])
    delete_all_files_in_directory(config["qdrant"]["storage_path"])

    create_collection.main()

    embedder = config["embedder"]
    embedder = get_embedder(embedder)

    qdrant_config = config["qdrant"]
    qdrant_client = instantiate_qclient(qdrant_config["storage_path"])

    parse_documents_as_json(
        config["paths"]["data"]["raw"],
        config["paths"]["data"]["preprocessed"],
        text_splitter=text_splitter,
    )

    update_document_jsons_with_embedding(
        config["paths"]["data"]["preprocessed"],
        config["paths"]["data"]["embedded"],
        embedder=embedder,
    )

    upload_document_jsons_to_qdrant(
        config["paths"]["data"]["embedded"],
        qdrant_collection=qdrant_config["document_collection"],
        qdrant_client=qdrant_client,
    )


if __name__ == "__main__":
    main()
