from langchain.chains import RetrievalQA

def build_retrieval_qa_chain(llm, retriever):
    '''
    Creates and returns an instance of a RetrievalQA chain given an llm and retriever

    RetrievalQA is a chain class designed to combine retrieval-based document search with QA via an LLM.
    Provided retriever fetches relevent content from knowledge base. Retrieved content is fed into LLM to generate a concise answer
    '''
    retrieval_qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        verbose=True
    )

    return retrieval_qa_chain