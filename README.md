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

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/Saransh2005/StudyAnalyzer.git
cd StudyAnalyzer
```

### 2. Create virtual environment & install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Add your Gemini API key
Create a `.env` file:
```
GOOGLE_API_KEY=your-gemini-api-key-here
```
Get your key at: https://aistudio.google.com/app/apikey

### 4. Set your PDF path
In `app.py`, update line 14:
```python
PDF_PATH = '/path/to/your/file.pdf'
```

### 5. Run
```bash
bash run.sh
```

## Example

```
PDF loaded. Ask questions (type 'exit' to quit).

Q: What are my skills?
A: Your skills include Python, React, LangChain, Docker...

Q: exit
```
