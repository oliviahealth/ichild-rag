import os
from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv
import psycopg2

import langchain
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI

from embeddings.openai import openai_embeddings
from vector_stores.pgvector import build_pg_vector_store
from chains.conversational_retrieval_chain_with_memory import build_conversational_retrieval_chain_with_memory

CORS()
load_dotenv()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
database_uri = os.getenv('DATABASE_URI')
langchain.verbose = True

# Connecting to postgresql database
db = SQLAlchemy(app)
with app.app_context():
    db.create_all()
conn = psycopg2.connect(database_uri)

# Build vector store and retriever
collection_name = "2024-09-02 00:44:49"
pg_vector_store = build_pg_vector_store(embeddings_model=openai_embeddings, collection_name=collection_name, connection_uri=database_uri)
pg_vector_retriever = pg_vector_store.as_retriever(search_type="mmr")

# Using OpenAI for LLM for now
llm = ChatOpenAI()

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/search")
def search():
    '''
    Example of basic RAG functionality. Route will take in a user query and pass it to a RetrievalQAChain.

    Final result will be sent to user
    '''
    search_query = request.args.get("query")

    retrieval_qa_chain = build_conversational_retrieval_chain_with_memory(llm, pg_vector_retriever)

    result = retrieval_qa_chain.run(search_query)

    return result