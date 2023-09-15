import os
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def ingest_docs() -> None:
    pdf_path = "pdfs/"
    loader = PyPDFDirectoryLoader(path=pdf_path)
    documents = loader.load()
    print(len(documents))
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(documents=documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    vectorstore.save_local("faiss_index_ludo")
    print(len(docs))


if __name__ == "__main__":
    print("hi")
    ingest_docs()

    # res = qa.run("Give me the best tactic to win in Ludo game")

    # print(res)
