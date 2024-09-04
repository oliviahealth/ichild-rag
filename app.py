import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv
import psycopg2

import langchain
from langchain_postgres import PGVector
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

CORS()
load_dotenv()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Connecting to postgresql database
db = SQLAlchemy(app)
with app.app_context():
    db.create_all()
conn = psycopg2.connect(os.getenv('DATABASE_URI'))

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

def test():
    langchain.verbose = True

    chat = ChatOpenAI()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings_model = OpenAIEmbeddings(openai_api_type=openai_api_key)

    collection_name = "2024-09-02 00:44:49"

    database_uri = os.getenv('DATABASE_URI')

    vector_store = PGVector(
        embeddings=embeddings_model,
        collection_name=collection_name,
        connection=database_uri,
        use_jsonb=True,
    )

    retriever = vector_store.as_retriever(search_type="mmr")

    chain = RetrievalQA.from_chain_type(
        llm=chat,
        retriever=retriever,
        chain_type="stuff",
        verbose=True
    )

    result = chain.run("When do women usually dialate during pregnancy?")
    
    print(result)

test()