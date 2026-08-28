"""
Personal PDF Chatbot - CLI Interface
Usage: python cli.py /path/to/your/file.pdf
"""

import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

if len(sys.argv) < 2:
    print("Usage: python cli.py /path/to/your/file.pdf")
    sys.exit(1)

PDF_PATH = sys.argv[1]
PERSIST_DIR = "chroma_db"


def build_qa_chain(pdf_path: str):
    print(f"[*] Loading PDF: {pdf_path}")
    docs = PyPDFLoader(pdf_path).load()
    print(f"[*] Loaded {len(docs)} pages. Chunking...")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    ).split_documents(docs)
    print(f"[*] Created {len(chunks)} chunks. Generating embeddings...")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectordb = Chroma.from_documents(
        chunks, embeddings, persist_directory=PERSIST_DIR
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

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
    print("\n✅ PDF loaded successfully! Ask questions (type 'exit' to quit).\n")
    while True:
        try:
            query = input("Q: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            answer = chain.invoke(query)
            print(f"\nA: {answer}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
