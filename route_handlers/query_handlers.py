import os
from uuid import uuid4

from chains.conversational_retrieval_chain_with_memory import build_conversational_retrieval_chain_with_memory
from langchain.chat_models import ChatOpenAI
from vector_stores.pgvector import build_pg_vector_store
from embeddings.openai import openai_embeddings

from retrievers.TableColumnRetriever import build_table_column_retriever

# Using OpenAI for LLM
llm = ChatOpenAI()

# Build vector store and retriever
collection_name = "2024-09-02 00:44:49"
pg_vector_store = build_pg_vector_store(
    embeddings_model=openai_embeddings, collection_name=collection_name, connection_uri=os.getenv('DATABASE_URI'))
pg_vector_retriever = pg_vector_store.as_retriever(search_type="mmr")

def search_direct_questions(id, search_query):
    '''
    Direct question handler searches OliviaHealth.org knowledge base for most relevant data relating to user query
    Data is passed to LLM to generate output
    Memory is updated with user query and answer

    Examples of direct questions: 'Newborn nutritonal advice', 'How do hormonal IUDs prevent pregnancy', 'What is mastitis treated with'
    '''

    if not id:
        id = uuid4()

    # Build the retrieval QA chain with SQL memory
    # Must pass in the session_id from the message_store table
    retrieval_qa_chain = build_conversational_retrieval_chain_with_memory(
        llm, pg_vector_retriever, id)

    result = retrieval_qa_chain.run(search_query)

    return result

def search_location_questions(id, search_query):
    '''
    Location question handler searches Locations table for most relevant locations relating to user query
    Data is converted to JSON array of locations
    Data is also passed to LLM to generate output
    Reponse includes the LLM response and the raw json array of locations
    Memory is updated with user query and answer

    Examples of location questions: 'Dental Services in Corpus Christi', 'Where can I get mental health support in Bryan'
    '''
    if not id:
        id = uuid4()

    locations = []
    
    # Creating a TableColumnRetriever to index all of the columns for the location table when retrieving documents
    table_column_retriever = build_table_column_retriever(
        connection_uri=os.getenv('DATABASE_URI'),
        table_name="location",
        column_names=["id", "name", "address", "city", "state", "country", "zip_code", "latitude", "longitude", "description", "phone", "sunday_hours", "monday_hours",
                      "tuesday_hours", "wednesday_hours", "thursday_hours", "friday_hours", "saturday_hours", "rating", "address_link", "website", "resource_type", "county"],
        embedding_column_name="embedding"
    )

    # Get the raw list of relevant locations
    doc_list = table_column_retriever.get_relevant_documents(search_query)

    # loop through the doc_list and for each doc add a json representation in the locations array
    for doc in doc_list:
        doc_id, name, address, city, state, country, zip_code, latitude, longitude, description, phone, sunday_hours, monday_hours, tuesday_hours, wednesday_hours, thursday_hours, friday_hours, saturday_hours, rating, address_link, website, resource_type, county = doc.page_content.split("##")

        unified_address = f"{address}, {city}, {state} {zip_code}"
        confidence = 1
        hours_of_operation = [{ "sunday": sunday_hours }, { "monday": monday_hours }, { "tuesday": tuesday_hours }, { "wednesday": wednesday_hours }, { "thursday": thursday_hours }, { "friday": friday_hours }, { "saturday": saturday_hours }]
        is_saved = False
        if(latitude and latitude.isnumeric()):
            latitude = float(latitude.strip())
        if(longitude and longitude.isnumeric()):
            longitude = float(longitude.strip())
        if(rating and rating.isnumeric()):
            rating = float(rating.strip())

        locations.append({
            "address": unified_address,
            "addressLink": address_link,
            "confidence" : confidence,
            "description" : description,
            "hoursOfOperation" : hours_of_operation,
            "id": doc_id,
            "isSaved": is_saved,
            "latitude": latitude,
            "longitude": longitude,
            "name": name,
            "phone": phone,
            "rating": rating,
            "website": website
        })

    # Using same conversational retrieval chain with SQL memory just with different retriever
    retrieval_qa_chain = build_conversational_retrieval_chain_with_memory(llm, table_column_retriever, id)

    # Get the LLM response
    response = retrieval_qa_chain.run(search_query)

    # Return the LLM response and the JSON
    return {
        "response" : response,
        "locations" : locations
    }


# Defining list of tools to use with OpenAI function calling
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