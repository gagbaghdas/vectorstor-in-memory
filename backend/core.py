import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS


def run_llm(query: str, chat_history: list[tuple[str, any]] = []) -> any:
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    new_vector_store = FAISS.load_local("faiss_index_ludo", embeddings=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=new_vector_store.as_retriever(),
        return_source_documents=True
    )

    return qa({"question": query, "chat_history": chat_history})
