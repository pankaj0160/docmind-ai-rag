# rag_pipeline.py

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["DISABLE_TF"] = "1"

import shutil
import gc
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from personality import get_personality_prompt

load_dotenv()

# ============================================================
# PATH CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_docs")

os.makedirs(UPLOAD_DIR, exist_ok=True)

COLLECTION_NAME = "docmind_collection"

# ============================================================
# GLOBAL RETRIEVER
# ============================================================

retriever = None
vector_store = None

# ============================================================
# EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully.")

# ============================================================
# LLM
# ============================================================

print("Loading LLM...")

llm = ChatMistralAI(
    model="mistral-large-latest"
)

print("LLM loaded successfully.")

# ============================================================
# SAVE UPLOADED FILES
# ============================================================

def save_uploaded_files(uploaded_files):
    saved_paths = []

    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, file.name)

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        saved_paths.append(file_path)

    return saved_paths


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents(file_paths):
    documents = []

    for path in file_paths:
        extension = os.path.splitext(path)[1].lower()

        if extension == ".pdf":
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

        elif extension == ".txt":
            loader = TextLoader(path, encoding="utf-8")
            documents.extend(loader.load())

    return documents


# ============================================================
# CHUNK DOCUMENTS
# ============================================================

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    return chunks


# ============================================================
# VECTOR STORE
# ============================================================

def create_or_update_vectorstore(chunks):
    global retriever
    global vector_store

    if not chunks:
        raise ValueError("No document chunks provided.")

    print("Creating fresh vector database...")

    temp_dir = tempfile.mkdtemp()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=temp_dir,
        collection_name=COLLECTION_NAME
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.7
        }
    )

    return vector_store


# ============================================================
# CLEAR DATABASE
# ============================================================

def clear_database():
    global retriever
    global vector_store

    try:
        retriever = None

        if vector_store:
            del vector_store
            vector_store = None

        gc.collect()

        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR, ignore_errors=True)

        os.makedirs(UPLOAD_DIR, exist_ok=True)

        return True

    except Exception as e:
        print("Clear DB Error:", e)
        return False


# ============================================================
# QUERY DOCUMENTS
# ============================================================

def query_documents(query, personality_mode):
    global retriever

    if retriever is None:
        raise ValueError("Please upload and process documents first.")

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    personality_prompt = get_personality_prompt(personality_mode)

    prompt = ChatPromptTemplate.from_template("""
You are an intelligent AI document assistant.

{personality_prompt}

Answer ONLY using the provided context.

If the answer is not found in the context, say:
"I could not find the answer in the uploaded documents."

Context:
{context}

Question:
{question}
""")

    final_prompt = prompt.invoke({
        "personality_prompt": personality_prompt,
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    return response.content, docs