"""Create a database ('Qdrant collection') for our chatbot."""

from rag_llm_applications.qdrant import create_db_collection, instantiate_qclient, delete_db_collection
from rag_llm_applications.util import PROJECT_ROOT, load_config, logging_setup


def main():
    config = load_config(PROJECT_ROOT / "config" / "config.yml")
    logging_setup(config)

    qdrant_config = config["qdrant"]
    qdrant_client = instantiate_qclient(qdrant_config["storage_path"])

    print("Deleting existing collection")
    delete_db_collection(qdrant_client, qdrant_config["document_collection"])

    print("Creating new collection")
    create_db_collection(
        qdrant_client, qdrant_config["document_collection"], qdrant_config["vectorsize"]
    )


if __name__ == "__main__":
    main()
