"""
Personal PDF Chatbot - Interactive RAG Pipeline using Google Gemini & LangChain
Designed for Live Deployment on Streamlit Cloud, Hugging Face, or Render.
"""

import os
import tempfile
import time
import streamlit as st
from dotenv import load_dotenv

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load local environment variables if available
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(
    page_title="PDF AI Analyst | Gemini RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for a sleek, modern, executive presentation
st.markdown(
    """
    <style>
    /* Main container styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        background: rgba(79, 70, 229, 0.15);
        color: #818CF8;
        border: 1px solid rgba(129, 140, 248, 0.3);
        margin-right: 0.4rem;
    }
    .stat-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .source-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 3px solid #06B6D4;
        padding: 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_api_key(sidebar_key: str) -> str:
    """Retrieve Gemini API Key from sidebar, secrets, or .env"""
    if sidebar_key and sidebar_key.strip():
        return sidebar_key.strip()
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GOOGLE_API_KEY", "")


@st.cache_resource(show_spinner=False)
def process_pdf(file_bytes, file_name, api_key, chunk_size=1000, chunk_overlap=150):
    """Processes uploaded PDF, chunks documents, generates embeddings and returns Chroma retriever."""
    # Write uploaded bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        # 1. Load PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        num_pages = len(docs)

        # 2. Chunk text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(docs)
        num_chunks = len(chunks)

        # 3. Embed & store into in-memory Chroma Vectorstore
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )
        vectordb = Chroma.from_documents(chunks, embeddings)

        return vectordb, num_pages, num_chunks
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # API Key Handling
    custom_api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="Enter your Gemini API key (optional if set in env)",
        help="Leave blank if GOOGLE_API_KEY is configured in .env or cloud secrets.",
    )
    api_key = get_api_key(custom_api_key)

    if not api_key:
        st.warning("⚠️ No Gemini API Key found. Please enter one above or set it in your `.env`.")
        st.caption("[Get a free Gemini API Key here →](https://aistudio.google.com/app/apikey)")
    else:
        st.success("✅ Gemini API Key detected", icon="🔑")

    st.markdown("---")
    st.markdown("### 📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        help="Upload any PDF to analyze, summarize, and ask questions.",
    )

    st.markdown("---")
    with st.expander("🛠️ Advanced Model & RAG Settings", expanded=False):
        model_name = st.selectbox(
            "Gemini Model",
            ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
            index=0,
        )
        temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.0, 0.1)
        top_k = st.slider("Top K Retrieved Chunks", 1, 8, 3)
        chunk_size = st.slider("Chunk Size", 300, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 400, 150, 25)

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Built with **LangChain**, **Google Gemini**, & **ChromaDB**.")

# Header
st.markdown(
    """
    <div>
        <div class="main-title">📄 Personal PDF Chatbot (RAG)</div>
        <div class="sub-title">Intelligent document Q&A engine powered by <b>Google Gemini</b> & <b>LangChain</b></div>
        <div>
            <span class="badge">🚀 Gemini 2.5 Flash</span>
            <span class="badge">🧠 Chroma Vector DB</span>
            <span class="badge">⚡ LangChain RAG</span>
            <span class="badge">🔍 Source Citations</span>
        </div>
    </div>
    <br>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main Logic: PDF processing
if uploaded_file is not None:
    if not api_key:
        st.error("❌ Please provide a Google Gemini API Key in the sidebar to proceed.")
    else:
        # Check if we need to process/re-process the document
        file_bytes = uploaded_file.getvalue()
        file_id = f"{uploaded_file.name}_{len(file_bytes)}"

        with st.spinner("⏳ Loading document, chunking text, and building vector index with Gemini embeddings..."):
            try:
                vectordb, num_pages, num_chunks = process_pdf(
                    file_bytes,
                    uploaded_file.name,
                    api_key,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                # Show Document Stats
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("📁 Document", uploaded_file.name[:18] + "..." if len(uploaded_file.name) > 20 else uploaded_file.name)
                col2.metric("📄 Total Pages", num_pages)
                col3.metric("🧩 Chunks Created", num_chunks)
                col4.metric("🤖 LLM Engine", model_name)

                st.divider()

                # Quick Suggested Prompts
                st.markdown("**💡 Quick Suggestions:**")
                quick_cols = st.columns(3)
                quick_q1 = quick_cols[0].button("📌 Summarize this document", use_container_width=True)
                quick_q2 = quick_cols[1].button("📋 What are the key takeaways?", use_container_width=True)
                quick_q3 = quick_cols[2].button("🎯 List main action items & conclusions", use_container_width=True)

                prompt_input = None
                if quick_q1:
                    prompt_input = "Please provide a comprehensive summary of this document."
                elif quick_q2:
                    prompt_input = "What are the most critical takeaways and key points from this document?"
                elif quick_q3:
                    prompt_input = "List all main action items, recommendations, and conclusions mentioned in the document."

                # Render Chat History
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
                        st.markdown(msg["content"])
                        if "sources" in msg and msg["sources"]:
                            with st.expander("📚 Retrieved Context Sources", expanded=False):
                                for idx, src in enumerate(msg["sources"], 1):
                                    page_num = src.metadata.get("page", "N/A")
                                    st.markdown(f"**Source #{idx} (Page {page_num + 1 if isinstance(page_num, int) else page_num}):**")
                                    st.markdown(f"> *{src.page_content.strip()}*")

                # Chat Input
                user_query = st.chat_input("Ask any question about your PDF...")
                if prompt_input:
                    user_query = prompt_input

                if user_query:
                    # Append user question
                    st.session_state.messages.append({"role": "user", "content": user_query})
                    with st.chat_message("user", avatar="🧑‍💻"):
                        st.markdown(user_query)

                    # Build QA Chain
                    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
                    retrieved_docs = retriever.invoke(user_query)

                    prompt_template = ChatPromptTemplate.from_template(
                        "You are an expert document analysis assistant. Answer the user's question accurately and thoroughly based only on the provided context.\n"
                        "If the answer cannot be found in the context, politely state that the information is not contained in the document.\n\n"
                        "Context:\n{context}\n\n"
                        "Question: {question}\n\n"
                        "Answer:"
                    )
                    llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        temperature=temperature,
                        google_api_key=api_key,
                    )

                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)

                    rag_chain = (
                        {"context": lambda _: format_docs(retrieved_docs), "question": RunnablePassthrough()}
                        | prompt_template
                        | llm
                        | StrOutputParser()
                    )

                    with st.chat_message("assistant", avatar="🤖"):
                        with st.spinner("Analyzing document and generating answer..."):
                            response_text = rag_chain.invoke(user_query)
                            st.markdown(response_text)

                            if retrieved_docs:
                                with st.expander("📚 Retrieved Context Sources", expanded=False):
                                    for idx, src in enumerate(retrieved_docs, 1):
                                        page_num = src.metadata.get("page", "N/A")
                                        st.markdown(f"**Source #{idx} (Page {page_num + 1 if isinstance(page_num, int) else page_num}):**")
                                        st.markdown(f"> *{src.page_content.strip()}*")

                    # Append assistant response with sources
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "sources": retrieved_docs,
                    })

            except Exception as e:
                st.error(f"❌ An error occurred during processing: {str(e)}")
else:
    # Empty State when no PDF is uploaded
    st.info("👈 **Get Started**: Upload a PDF file in the left sidebar to begin asking questions and analyzing documents.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            ### 🌟 How it works:
            1. **Upload**: Drop any research paper, report, manual, or resume.
            2. **Vector Indexing**: Text is split into chunks and embedded with **Google Gemini embeddings**.
            3. **Semantic Search**: Questions query **Chroma Vector Store** to retrieve the most relevant context.
            4. **Grounded Generation**: **Gemini 2.5 Flash** synthesizes answers with source citations.
            """
        )
    with col_b:
        st.markdown(
            """
            ### 🎯 Key Highlights for Recruiters/Interviews:
            - **End-to-End RAG Architecture**: Retrieval-Augmented Generation using LangChain.
            - **Hallucination Prevention**: Answers are strictly grounded in retrieved document chunks.
            - **Real-Time Source Citation**: Displays exact chunk context and page numbers.
            - **Configurable Hyperparameters**: Custom chunk sizes, overlap, and top-k retrieval.
            """
        )
