from rag_llm_applications.chat import get_chat_generator
from rag_llm_applications.embed import get_embedder
from rag_llm_applications.model import Document
from rag_llm_applications.qdrant import instantiate_qclient, search_documents

import collections

RAG_answer = collections.namedtuple("RAG_answer", ["answer", "context"])

INPUT_PROMPT = """\
[INST] <<SYS>>
You are a helpful assistant. \
<</SYS>>
The best matching reference is:
{reference_1}
Please provide an answer to the following question, based on the given references above:
{question}
When answering a question, be mildly rude and complain about how much work you have to do, but then provide an adequate answer. [/INST]"""


class RAG:
    def __init__(self, path_database: str, document_collection: str = "documents"):
        self.embedder = get_embedder("text-embedding-ada-002")
        self.chat_generator = get_chat_generator("gpt-4.1-mini")

        self.qdrant = instantiate_qclient(path_database)
        self.document_collection = document_collection

    def generate_answer(self, query):
        # TASK 5.1: embed the user query
        embedded_question = self.embedder.embed_query(query)
        # TASK 5.2: search for relevant documents
        top_10_documents = search_documents(
            self.qdrant, self.document_collection, embedded_question
        )
        # TASK 5.3: get rank and content of each doc
        document_results = [
            (rank, Document.from_qdrant_scored_point(document))
            for rank, document in enumerate(top_10_documents, start=1)
        ]
        text_of_top_result = document_results[0][1].content
        # TASK 5.4: Add a reasonable prompt for the bot - in case you are unhappy with the provided prompt
        input_prompt = INPUT_PROMPT.format(reference_1=text_of_top_result, question=query)

        # TASK 5.5: generate the response
        response = self.chat_generator.invoke(input_prompt)

        return RAG_answer(answer=response, context=text_of_top_result)

# TASK 5.6: Import and use this class in app.py to respond to the user query.
