# StudyAnalyzer — AI-Powered PDF Chatbot (RAG + Google Gemini)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-v1.0+-green.svg)](https://www.langchain.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-orange.svg)](https://ai.google.dev/)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-purple.svg)](https://www.trychroma.com/)

An end-to-end AI document intelligence platform built with **Retrieval-Augmented Generation (RAG)** using **Google Gemini**, **LangChain**, and **ChromaDB**. Upload any document (research papers, reports, resumes, manuals) and query it in real-time with grounded citations and source verification.

---

## 🌟 Key Features

- **Interactive Web Interface**: Built with Streamlit for seamless drag-and-drop document upload and conversational interaction.
- **RAG Architecture**: Prevents LLM hallucinations by retrieving exact vector chunks before generating answers.
- **Real-Time Source Citations**: Shows chunk snippets and page numbers for transparent answers.
- **Configurable Hyperparameters**: Custom controls for chunk size, chunk overlap, top-K retrieval, temperature, and Gemini model version.
- **CLI & Web Support**: Run via terminal or as a full web app.
- **1-Click Free Cloud Deployment**: Fully ready for Streamlit Community Cloud and Hugging Face Spaces.

---

## 🏗️ Architecture Flow

```text
[PDF Upload] ──► [PyPDFLoader] ──► [RecursiveCharacterTextSplitter]
                                                │
                                                ▼
                                [Gemini Embedding 001]
                                                │
                                                ▼
                                    [ChromaDB Vector Store]
                                                │
[User Question] ──► [Semantic Similarity Search] (Top-K Chunks)
                                                │
                                                ▼
                         [Gemini 2.5 Flash LLM + Prompt Template]
                                                │
                                                ▼
                             [Grounded Answer + Page Sources]
```

---

## 🚀 Quick Start (Local)

### 1. Clone & Setup
```bash
git clone https://github.com/Saransh2005/StudyAnalyzer.git
cd StudyAnalyzer
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Configure API Key
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```
*(Get a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey))*

### 4. Run the Application

**Option A: Modern Web UI (Recommended)**
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

**Option B: Terminal CLI**
```bash
./run.sh /path/to/your/file.pdf
```
or
```bash
python cli.py /path/to/your/file.pdf
```

---

## 🌐 Free Live Cloud Deployment (Streamlit Community Cloud)

You can deploy this project live with a public URL in **2 minutes for free**:

1. Push your latest code to GitHub:
   ```bash
   git add .
   git commit -m "Add interactive Streamlit Web UI and RAG pipeline"
   git push origin main
   ```
2. Visit **[share.streamlit.io](https://share.streamlit.io/)** and sign in with your GitHub account.
3. Click **"New app"** and select:
   - **Repository**: `Saransh2005/StudyAnalyzer`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. In **Advanced Settings → Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your_actual_gemini_api_key"
   ```
5. Click **Deploy!** Your app will be live at `https://<your-app-name>.streamlit.app`! 🎉

---

## 🛠️ Tech Stack

- **Large Language Model**: Google Gemini 2.5 Flash
- **Embeddings**: Google Gemini Embedding 001
- **Vector Database**: ChromaDB
- **Orchestration**: LangChain LCEL (LangChain Expression Language)
- **Document Ingestion**: LangChain PyPDFLoader & Character Text Splitters
- **Frontend / Deployment**: Streamlit
