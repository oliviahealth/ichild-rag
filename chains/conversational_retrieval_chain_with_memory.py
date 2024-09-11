from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.file import FileChatMessageHistory
from langchain.chains import ConversationalRetrievalChain

def build_conversational_retrieval_chain_with_memory(llm, retriever):
    memory = ConversationBufferMemory(
        chat_memory=FileChatMessageHistory("messages.json"),
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
