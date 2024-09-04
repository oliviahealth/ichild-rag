import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv
import psycopg2

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