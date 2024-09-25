from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import UserDefinedType
from uuid import uuid4

db = SQLAlchemy()

class Vector(UserDefinedType):
    def get_col_spec(self):
        return "VECTOR(1536)"

    def bind_expression(self, bindvalue):
        return bindvalue

    def column_expression(self, col):
        return col

class Location(db.Model):
    id = db.Column(db.String(), primary_key=True, default=lambda: str(uuid4()))
    name = db.Column(db.String(), nullable=False, unique=True)
    address = db.Column(db.String(), nullable=False)
    city = db.Column(db.String(), nullable=False)
    state = db.Column(db.String(), nullable=False)
    country = db.Column(db.String(), nullable=False)
    zip_code = db.Column(db.String(), nullable=False)
    county = db.Column(db.String(), nullable=False)
    latitude = db.Column(db.Float(), nullable=False)
    longitude = db.Column(db.Float(), nullable=False)
    description = db.Column(db.String(), nullable=False)
    phone = db.Column(db.String(), nullable=False)
    sunday_hours = db.Column(db.String(), nullable=False)
    monday_hours = db.Column(db.String(), nullable=False)
    tuesday_hours = db.Column(db.String(), nullable=False)
    wednesday_hours = db.Column(db.String(), nullable=False)
    thursday_hours = db.Column(db.String(), nullable=False)
    friday_hours = db.Column(db.String(), nullable=False)
    saturday_hours = db.Column(db.String(), nullable=False)
    rating = db.Column(db.String(), nullable=False)
    address_link = db.Column(db.String(), nullable=False)
    website = db.Column(db.String(), nullable=False)
    resource_type = db.Column(db.String(), nullable=False)
    embedding = db.Column(Vector(), nullable=True)