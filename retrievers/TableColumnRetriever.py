import ast
from typing import List
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from psycopg2 import connect
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import OpenAIEmbeddings

class TableColumnRetriever(BaseRetriever):
    """A retriever that retrieves top-k documents for a given table and its columns based on OpenAI embedding similarity."""

    documents: List[Document]
    embeddings: List[np.ndarray]
    k: int
    openai_embeddings: OpenAIEmbeddings
    """Number of top results to return."""

    def get_relevant_documents(
        self, query: str
    ) -> List[Document]:
        """Retrieve documents based on cosine similarity between embeddings."""

        # Step 1: Convert the query into an embedding using OpenAI
        query_embedding = self.openai_embeddings.embed_query(query)

        # Step 2: Compute cosine similarities between query and document embeddings
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        # Step 3: Get the top-k most similar documents
        top_k_indices = np.argsort(similarities)[-self.k:][::-1]

        matching_documents = [self.documents[i] for i in top_k_indices]

        return matching_documents


def build_table_column_retriever(connection_uri, table_name, column_names, embedding_column_name):
    conn = connect(connection_uri)
    cursor = conn.cursor()

    # Fetch embeddings and content from the database
    columns_str = ', '.join(column_names)
    cursor.execute(f"SELECT {columns_str}, {embedding_column_name} FROM {table_name};")

    rows = cursor.fetchall()

    documents = [
        Document(page_content="##".join([str(row[i]) for i in range(len(column_names))]))
        for row in rows
    ]

    embeddings = [np.array(ast.literal_eval(row[len(column_names)])) for row in rows]

    # Initialize OpenAIEmbeddings from LangChain
    openai_embeddings = OpenAIEmbeddings()

    # Create the retriever with OpenAI embeddings
    retriever = TableColumnRetriever(documents=documents, embeddings=embeddings, k=4, openai_embeddings=openai_embeddings)

    return retriever