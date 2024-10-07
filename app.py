import os
from flask import Flask, request
from flask_cors import CORS
from dotenv import load_dotenv
from uuid import uuid4
import psycopg2
import langchain
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI

from embeddings.openai import openai_embeddings
from vector_stores.pgvector import build_pg_vector_store
from chains.conversational_retrieval_chain_with_memory import build_conversational_retrieval_chain_with_memory
from database.database import db
from retrievers.TableColumnRetriever import build_table_column_retriever

from functools import partial
from typing import Literal, TypedDict
import json
from openai import OpenAI

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

# Using OpenAI for LLM
llm = ChatOpenAI()

class QueryType(TypedDict):
    query_type: Literal["general", "location"]
    reasoning: str


def determine_query_type(client: OpenAI, query: str) -> QueryType:
    functions = [
        {
            "name": "classify_query",
            "description": "Classify whether a query requires location-based information or general information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["general", "location"],
                        "description": "The type of query - 'location' for queries needing location-based responses, 'general' for other queries"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "A brief explanation of why this classification was chosen"
                    }
                },
                "required": ["query_type", "reasoning"]
            }
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that classifies user queries."},
            {"role": "user", "content": f"Classify this query: {query}"}
        ],
        functions=functions,
        function_call={"name": "classify_query"}
    )
    
    return json.loads(response.choices[0].message.function_call.arguments)

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

@app.route("/test/", defaults={"id": None})
@app.route("/test/<id>")
def test(id):
    table_column_retriever = build_table_column_retriever(
        connection_uri=database_uri, 
        table_name="location", 
        column_names=["name", "address", "city", "state", "country", "zip_code", "latitude", "longitude", "description", "phone", "sunday_hours", "monday_hours", "tuesday_hours", "wednesday_hours", "thursday_hours", "friday_hours", "saturday_hours", "rating", "address_link", "website", "resource_type", "county"], 
        embedding_column_name="embedding"
    )

    search_query = request.args.get("query")
    # documents = table_column_retriever.get_relevant_documents(query)

    query_classification = determine_query_type(OpenAI(), search_query)
    print(query_classification)

    if not id:
        id = uuid4()

    retrieval_qa_chain = build_conversational_retrieval_chain_with_memory(llm, table_column_retriever, id)

    result = retrieval_qa_chain.run(search_query)

    return result

if __name__ == '__main__':
    app.run(debug=True)