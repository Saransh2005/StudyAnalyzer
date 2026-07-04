
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # expects GOOGLE_API_KEY in .env

PDF_PATH = '/Users/saranshsingh8888icloud.com/Desktop/saransh/untitled folder/saranshD.pdf'
PERSIST_DIR = "chroma_db"


def build_qa_chain(pdf_path: str):
    # 1. Load PDF
    docs = PyPDFLoader(pdf_path).load()

    # 2. Split into chunks
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    ).split_documents(docs)

    # 3. Embed + store in Chroma using Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectordb = Chroma.from_documents(
        chunks, embeddings, persist_directory=PERSIST_DIR
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # 4. Build RAG chain with Gemini LLM
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the following context:\n\n"
        "{context}\n\n"
        "Question: {question}"
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def main():
    chain = build_qa_chain(PDF_PATH)
    print("PDF loaded. Ask questions (type 'exit' to quit).")
    while True:
        query = input("\nQ: ")
        if query.lower() == "exit":
            break
        answer = chain.invoke(query)
        print("A:", answer)


if __name__ == "__main__":
    main()
