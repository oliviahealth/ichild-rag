from typing import List
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from psycopg2 import connect

class TableColumnRetriever(BaseRetriever):
    """A retriever that contains the top k documents that contain the user query for a given table and column.

    This retriever only implements the sync method _get_relevant_documents.

    If the retriever were to involve file access or network access, it could benefit
    from a native async implementation of `_aget_relevant_documents`.
    """

    documents: List[Document]
    """List of documents to retrieve from."""
    k: int
    """Number of top results to return"""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Sync implementations for retriever."""
        matching_documents = []
        for document in self.documents:
            if len(matching_documents) > self.k:
                return matching_documents

            if query.lower() in document.page_content.lower():
                matching_documents.append(document)
        return matching_documents


def build_column_retriever(connection_uri, table_name, column_name):
    conn = connect(connection_uri)
    cursor = conn.cursor()

    cursor.execute(f"SELECT {column_name} FROM {table_name};")

    rows = cursor.fetchall()

    documents = [
        Document(page_content=row[0])
        for row in rows
    ]

    retriever = TableColumnRetriever(documents=documents, k=4)

    return retriever