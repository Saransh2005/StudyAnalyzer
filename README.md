# StudyAnalyzer — PDF Chatbot (RAG + Gemini)

An AI-powered PDF question-answering chatbot built with a **Retrieval-Augmented Generation (RAG)** pipeline using **Google Gemini** and **ChromaDB**.

## How it works

```
PDF → Chunk → Embed (Gemini) → Store (ChromaDB)
                                      ↓
Question → Embed → Similarity Search → Top 3 Chunks → Gemini LLM → Answer
```

## Tech Stack

- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: Gemini Embedding 001
- **Vector DB**: ChromaDB
- **Orchestration**: LangChain LCEL
- **PDF Loader**: LangChain PyPDFLoader

---

## Setup & Run

### Step 1 — Clone the repo
```bash
git clone https://github.com/Saransh2005/StudyAnalyzer.git
cd StudyAnalyzer
```

### Step 2 — Create virtual environment & install dependencies
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### Step 3 — Get a Gemini API key (free)
1. Go to https://aistudio.google.com/app/apikey
2. Click **Create API Key**
3. Create a `.env` file in the project folder:
```
GOOGLE_API_KEY=your-api-key-here
```

### Step 4 — Run with your PDF
```bash
bash run.sh /path/to/your/file.pdf
```

Or directly with Python:
```bash
python app.py /path/to/your/file.pdf
```

---

## Example

```
PDF loaded. Ask questions (type 'exit' to quit).

Q: What are my skills?
A: Your skills include Python, React, LangChain, Docker...

Q: Summarize this document
A: This document is a resume for...

Q: exit
```

---

## Notes
- The first run takes ~30 seconds to embed the PDF into ChromaDB
- Subsequent runs reuse the cached `chroma_db/` folder (faster)
- Only answers based on the content in your PDF — no hallucinations from outside data
