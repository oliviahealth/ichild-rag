import os
import glob
from datetime import datetime
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv
import psycopg2

from langchain.embeddings import OpenAIEmbeddings

from preprocessing.load_kb import load_kb

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

# Using OpenAI embeddings for now
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/load-kb")
def appRoute():
    '''
    Loads knowledge base into vector database by calling load_kb().

    Must specify knowledge base path (kb_path), collection name (collection_name) and database uri (database_uri)

    Returns 200 OK on success

    Note: This function calls OpenAIEmbeddings() which costs money to run and can be fairly expensive so try to limit this operation.
          Ideally, the vector database should only need to be loaded initially and whenever we have new data
    '''

    kb_path = "./knowledge_base/"
    collection_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    database_uri = os.getenv("DATABASE_URI")
    
    load_kb(embeddings_model=embeddings_model, documents_path=kb_path, collection_name=collection_name, database_uri=database_uri)

    return "OK"