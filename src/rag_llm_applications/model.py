"""In this code, we define the Document class, which is used to represent a document in our system. 
It specifies key attributes of a document that we will work with, such as its content, source and page number."""
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, TypedDict

logger = logging.getLogger("chatbot_in_a_day")


class LangchainDocument(Protocol):
    page_content: str
    metadata: dict[str, str | int]


class PayloadDict(TypedDict):
    content: str
    source: str
    page: int


class ScoredPoint(Protocol):
    payload: PayloadDict
    score: float


@dataclass
class Document:
    content: str
    source: str
    page: int
    embedding: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.name = Path(self.source).stem

    def add_embedding(self, embedding: list[float]):
        self.embedding = embedding

    @staticmethod
    def from_langchain_document(document: LangchainDocument):
        """Here, we extract the content, source and page from the document"""
        return Document(
            content=document.page_content,
            source=document.metadata["source"],
            page=document.metadata["page"]
        )

    @staticmethod
    def from_json_file(file_name: str | Path):
        """Here, we load the (preprocessed) document from a json file"""
        with open(file_name, "r", encoding="utf-8") as json_data:
            logger.debug("Loading document from json %s", file_name)
            document_data = json.load(json_data)
            return Document(**document_data)

    @staticmethod
    def from_qdrant_scored_point(scored_point: ScoredPoint):
        """Here, we load the document from a scored point, which is the result of a search query"""
        return Document(
            content=scored_point.payload["content"],
            source=scored_point.payload["source"],
            page=scored_point.payload["page"]
        )

    def save(self, target_dir: str | Path):
        """Save the document to a json file"""
        save_location = Path(target_dir)
        save_location.mkdir(parents=True, exist_ok=True)
        file_name = save_location / f"{self.name}_{self.page}.json"
        logger.debug("Saving document to %s", file_name)

        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=4)
