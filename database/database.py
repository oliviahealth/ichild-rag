from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSONB
from uuid import uuid4

db = SQLAlchemy()

# Define the Conversation model
# class Conversation(db.Model):
#     __tablename__ = 'conversations'

#     id = db.Column(db.String, primary_key=True)
#     messages = db.Column(db.ARRAY(JSONB), default=[], nullable=False)