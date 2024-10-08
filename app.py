import os
from flask import Flask, request
from flask_cors import CORS
from dotenv import load_dotenv
import psycopg2
import langchain
from database.database import db
import openai

from route_handlers.query_handlers import search_direct_questions, search_location_questions, tools

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

# Basic hello world route
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/search/", defaults={"id": None})
@app.route("/search/<id>")
def unified_search(id):
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
