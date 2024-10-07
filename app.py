import os
import json
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
import openai

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
pg_vector_store = build_pg_vector_store(
    embeddings_model=openai_embeddings, collection_name=collection_name, connection_uri=database_uri)
pg_vector_retriever = pg_vector_store.as_retriever(search_type="mmr")

# Using OpenAI for LLM
llm = ChatOpenAI()

functions = [
    {
        "name": "direct_answer_query",
        "description": "Handles direct queries like 'newborn nutritional advice' or general questions expecting factual responses.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user query for factual information.",
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "location_based_query",
        "description": "Handles location-based queries like 'Where can I find mental health support in Bryan, Texas' or 'Find a nearby hospital'.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user query for finding locations.",
                }
            },
            "required": ["query"]
        }
    }
]

# Helper function to determine the appropriate query type using OpenAI function calling


def classify_query_type(query):
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Use a model supporting function calling
        messages=[
            {
                "role": "system",
                "content": "You are an intelligent assistant that routes user queries to the correct function based on the query type."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        functions=functions,
        function_call="auto"
    )
    # Extract the name of the function the model decided to call
    function_call = response["choices"][0]["message"]["function_call"]["name"]
    return function_call

# Basic hello world route


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

# Search route with conversation ID


def search_direct_questions(id, search_query):
    '''
    Example of basic RAG functionality. Route will take in a user query and pass it to a RetrievalQAChain.
    Final result will be sent to the user.
    '''
    search_query = request.args.get("query")

    if not id:
        id = uuid4()

    # Build the retrieval QA chain with SQL memory
    # Must pass in the session_id from the message_store table
    retrieval_qa_chain = build_conversational_retrieval_chain_with_memory(
        llm, pg_vector_retriever, id)

    result = retrieval_qa_chain.run(search_query)

    return result


def search_location_questions(id, search_query):
    table_column_retriever = build_table_column_retriever(
        connection_uri=database_uri,
        table_name="location",
        column_names=["name", "address", "city", "state", "country", "zip_code", "latitude", "longitude", "description", "phone", "sunday_hours", "monday_hours",
                      "tuesday_hours", "wednesday_hours", "thursday_hours", "friday_hours", "saturday_hours", "rating", "address_link", "website", "resource_type", "county"],
        embedding_column_name="embedding"
    )

    # documents = table_column_retriever.get_relevant_documents(query)

    if not id:
        id = uuid4()

    retrieval_qa_chain = build_conversational_retrieval_chain_with_memory(
        llm, table_column_retriever, id)

    result = retrieval_qa_chain.run(search_query)

    return result


tools = [
    {
        "type": 'function',
        "function": {
            "name": "search_direct_questions",
            "description": "Retrieve a direct answer from the knowlege base based on a user question. Call this whenever you get a direct question that should be answer without a specific location. For example when a user asks 'newborn nutritional advice' or 'birth control alternatives'",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The conversation id. For new conversations, this will be null, however for existing conversations, this will be passed in by the user to continue that conversation",
                    },
                    "query": {
                        "type": "string",
                        "description": "The question the user is trying to find an answer for"
                    }
                },
                "required": ["id", "query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": 'function',
        "function": {
            "name": "search_location_questions",
            "description": "Retrieve a location from the locations table based on a user question. Call this whenever you get a question that should be answer with a specific location. For example when a user asks 'mental health support in Bryan, Texas' or 'Where can i get a root canal in Corpus Christi'",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The conversation id. For new conversations, this will be null, however for existing conversations, this will be passed in by the user to continue that conversation",
                    },
                    "query": {
                        "type": "string",
                        "description": "The question the user is trying to find an answer for"
                    }
                },
                "required": ["id", "query"],
                "additionalProperties": False
            }
        }
    }
]


@app.route("/search/", defaults={"id": None})
@app.route("/search/<id>")
def test(id):
    search_query = request.args.get("query")

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the supplied tools to assist the user."},
        {"role": "user", "content": search_query}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )

    refusal = response.choices[0].message.refusal

    if(refusal):
        return "Something went wrong: OpenAi Classification Refusal"

    function_name = response.choices[0].message.tool_calls[0].function.name

    data = None
    if(function_name == 'search_direct_questions'):
        data = search_direct_questions(id, search_query)
    elif(function_name == 'search_location_questions'):
        data = search_location_questions(id, search_query)
    else:
        return "error"

    return data

if __name__ == '__main__':
    app.run(debug=True)
