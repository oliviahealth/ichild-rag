from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
import os

def build_conversational_retrieval_chain_with_memory(llm, retriever, id):
    # Memory is stored and sourced from SQL
    # All messages (Human and AI) are stored in the message_store table and are linked together via the session_id
    # To continue an existing conversation, pass in an existing session_id
    # To create a new conversation, pass in a new session_id
    memory = ConversationBufferMemory(
        chat_memory=SQLChatMessageHistory(session_id=str(id), connection_string=os.getenv('DATABASE_URI')),
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        condense_question_llm=llm
    )