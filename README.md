# 🧠 DocMind AI

<div align="center">

### AI-Powered Document Question Answering Assistant using RAG

Ask intelligent questions from your uploaded documents using **Retrieval-Augmented Generation (RAG)** powered by **LangChain**, **ChromaDB**, **HuggingFace Embeddings**, **Mistral AI**, and **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Database-purple?style=for-the-badge)
![Mistral AI](https://img.shields.io/badge/Mistral-AI-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red?style=for-the-badge&logo=streamlit)

</div>

---

# 📌 Overview

DocMind AI is an intelligent **Document AI Assistant** that allows users to upload documents and ask natural language questions based on their content.

Instead of relying only on the language model’s pre-trained knowledge, the system uses **Retrieval-Augmented Generation (RAG)** to fetch relevant document chunks and generate grounded answers.

This ensures:

✅ More accurate responses  
✅ Context-aware answers  
✅ Reduced hallucination  
✅ Support for custom private documents  

---

# 🚀 Features

## 📂 Document Upload Support

Upload:

- PDF documents
- TXT files

---

## 🧠 RAG Pipeline

Implements complete Retrieval-Augmented Generation workflow:

- Document Loading
- Text Chunking
- Embedding Generation
- Vector Storage
- Semantic Retrieval
- Context Augmentation
- LLM Response Generation

---

## 🎭 Personality Modes

Interact with the assistant in different personalities:

- Professional
- Friendly
- Teacher
- Motivational
- Concise

---

## 💾 Persistent Vector Database

Uses **ChromaDB** to persist embeddings locally.

Benefits:

- Faster repeated querying
- No need to reprocess documents every time
- Local vector storage support

---

## 🔍 Semantic Search

Instead of keyword matching, DocMind AI uses embedding similarity search for smarter retrieval.

---

## 🤖 Mistral AI Integration

Uses **Mistral Large** for intelligent answer generation.

---

# 🏗️ System Architecture

```text
                +----------------------+
                |   User Uploads File  |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Document Loader      |
                | (PDF / TXT)          |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Text Chunking        |
                | Recursive Splitter   |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Embedding Model      |
                | HuggingFace          |
                | all-MiniLM-L6-v2     |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Chroma Vector DB     |
                +----------+-----------+
                           |
        User Query         |
            |              |
            v              |
 +-------------------+     |
 | Query Embedding   |-----+
 +---------+---------+
           |
           v
 +----------------------+
 | Semantic Retrieval   |
 | Top Relevant Chunks  |
 +----------+-----------+
            |
            v
 +----------------------+
 | Prompt Augmentation  |
 +----------+-----------+
            |
            v
 +----------------------+
 | Mistral AI LLM       |
 +----------+-----------+
            |
            v
 +----------------------+
 | Final Response       |
 +----------------------+
```

---

# 🛠️ Tech Stack

| Technology | Purpose |
|---------|---------|
| Python | Backend Development |
| Streamlit | Frontend UI |
| LangChain | RAG Orchestration |
| ChromaDB | Vector Database |
| HuggingFace Embeddings | Text Embeddings |
| Mistral AI | LLM Generation |
| PyPDF | PDF Processing |
| dotenv | Environment Variables |

---

# 📁 Project Structure

```bash
DocMind-AI/
│
├── app.py
├── rag_pipeline.py
├── personality.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
│
├── uploaded_docs/
│
└── chroma_db/
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/docmind-ai-rag.git
```

```bash
cd docmind-ai-rag
```

---

## 2. Create Virtual Environment

Windows:

```bash
python -m venv rag_env
```

Activate:

```bash
rag_env\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Setup

Create `.env`

```env
MISTRAL_API_KEY=your_api_key_here
```

---

# ▶️ Run Application

```bash
python -m streamlit run app.py
```

Application opens at:

```bash
http://localhost:8501
```

---

# 🧪 How RAG Works in This Project

Example query:

> "Why is DSA difficult?"

Flow:

1. User asks question
2. Query converted to embedding
3. Chroma searches relevant document chunks
4. Matching chunks retrieved
5. Context injected into prompt
6. Mistral generates grounded answer

This is **true RAG architecture**.

---

# 📸 Example Workflow

### Upload Document

- DSA Guide PDF
- Research Notes
- TXT Knowledge Base

### Ask Questions

Examples:

```text
What is dynamic programming?
Why is DSA difficult?
Explain DFS with examples.
Summarize chapter 3.
```

---

# 🔍 Debug Verification

Terminal logs show:

```text
RAG PIPELINE STARTED
Loading vector database...
Retriever created.
Searching relevant chunks...
Retrieved 4 chunks
Sending retrieved context to LLM...
LLM response generated.
```

Confirms:

✅ Retrieval  
✅ Augmentation  
✅ Generation  

---

# 🚧 Future Enhancements

Planned improvements:

- DOCX support
- Multiple PDF collections
- Source citation UI
- Chat history
- Authentication
- Cloud deployment
- OCR support
- Voice interaction
- Multi-agent workflows
- Hybrid retrieval (BM25 + Vector Search)

---

# 🧠 Learning Concepts Implemented

This project demonstrates:

- RAG Architecture
- Semantic Search
- Vector Databases
- Embeddings
- Prompt Engineering
- LLM Integration
- Streamlit App Development
- Environment Variable Security
- File Handling
- Persistent Storage

---

# 👨‍💻 Author

## Pankaj Thakur

AI & Data Science Student  
Passionate about Generative AI, RAG Systems, AI Engineering, and Full Stack Development

---

# 📜 License

This project is created for learning and educational purposes.

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

</div>
