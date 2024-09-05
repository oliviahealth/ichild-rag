from langchain_postgres import PGVector

def build_pg_vector_store(embeddings_model, collection_name, connection_uri):
    '''
    Builds and returns an instance of a PGVector store given an embeddings model, db collection name and db connection url.

    Built instance can be used for semantic search and retrieval functionality
    '''
    vector_store = PGVector(
        embeddings=embeddings_model,
        collection_name=collection_name,
        connection=connection_uri,
        use_jsonb=True,
    )

    return vector_store