import os
from flask import Flask, request
from flask_cors import CORS
from dotenv import load_dotenv
from uuid import uuid4
import psycopg2
import langchain
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI

from typing import List
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from psycopg2 import connect

from embeddings.openai import openai_embeddings
from vector_stores.pgvector import build_pg_vector_store
from chains.conversational_retrieval_chain_with_memory import build_conversational_retrieval_chain_with_memory
from database.database import db, Location

load_dotenv()

app = Flask(__name__)
CORS(app)

# Load database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
database_uri = os.getenv('DATABASE_URI')

# Creating all tables
with app.app_context():
    db.init_app(app)  
    db.create_all()

# Establish connection to PostgreSQL (optional, if you need to use raw psycopg2)
conn = psycopg2.connect(database_uri)

# Enable verbose logging in LangChain
langchain.verbose = True

# Build vector store and retriever
collection_name = "2024-09-02 00:44:49"
pg_vector_store = build_pg_vector_store(embeddings_model=openai_embeddings, collection_name=collection_name, connection_uri=database_uri)
pg_vector_retriever = pg_vector_store.as_retriever(search_type="mmr")

def fetch_documents_from_db() -> list:
    """Fetch descriptions from the locations table and convert them into Document objects."""
    
    locations = Location.query.all()
        
    # Step 3: Convert each description into a Document object
    documents = [
        Document(page_content=loc.description, metadata={"name": loc.name, "address": loc.address},) for loc in locations
    ]
    
    return documents

class ToyRetriever(BaseRetriever):
    """A toy retriever that contains the top k documents that contain the user query.

    This retriever only implements the sync method _get_relevant_documents.

    If the retriever were to involve file access or network access, it could benefit
    from a native async implementation of `_aget_relevant_documents`.

    As usual, with Runnables, there's a default async implementation that's provided
    that delegates to the sync implementation running on another thread.
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
    
# Using OpenAI for LLM
llm = ChatOpenAI()

# Basic hello world route
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

# Search route with conversation ID
@app.route("/search/", defaults={"id": None})
@app.route("/search/<id>")
def search(id):
    '''
    Example of basic RAG functionality. Route will take in a user query and pass it to a RetrievalQAChain.
    Final result will be sent to the user.
    '''
    search_query = request.args.get("query")

    if not id:
        id = uuid4()

    # Build the retrieval QA chain with SQL memory
    # Must pass in the session_id from the message_store table
    retrieval_qa_chain = build_conversational_retrieval_chain_with_memory(llm, pg_vector_retriever, id)

    result = retrieval_qa_chain.run(search_query)

    return result

@app.route("/test")
def test():

    documents = fetch_documents_from_db()
    
    retriever = ToyRetriever(documents=documents, k=4)

    t = retriever.invoke("Corpus Christi Women's Clinic")

    print(t)

    return "success"

if __name__ == '__main__':
    app.run(debug=True)