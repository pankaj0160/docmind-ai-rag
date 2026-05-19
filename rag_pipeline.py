# rag_pipeline.py

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["DISABLE_TF"] = "1"

import shutil
import gc
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
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

COLLECTION_NAME = "docmind_collection"

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
    """
    Save uploaded files locally.
    """

    saved_paths = []

    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, file.name)

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        print(f"Saved file: {file.name}")
        saved_paths.append(file_path)

    return saved_paths


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents(file_paths):
    """
    Load PDF and TXT files.
    """

    documents = []

    for path in file_paths:
        extension = os.path.splitext(path)[1].lower()

        if extension == ".pdf":
            print(f"Loading PDF: {path}")
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

        elif extension == ".txt":
            print(f"Loading TXT: {path}")
            loader = TextLoader(path, encoding="utf-8")
            documents.extend(loader.load())

    print(f"Total documents loaded: {len(documents)}")

    return documents


# ============================================================
# CHUNK DOCUMENTS
# ============================================================

def split_documents(documents):
    """
    Split documents into chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    if chunks:
        print("\n===== SAMPLE CHUNK =====")
        print(chunks[0].page_content[:500])
        print("========================\n")

    return chunks


# ============================================================
# VECTOR STORE
# ============================================================

def create_or_update_vectorstore(chunks):
    """
    Create fresh vector database.
    Old database is replaced completely.
    """

    print("Creating fresh vector database...")

    vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME
    )

    vector_store.add_documents(chunks)

    print("Vector database created successfully.")

    del vector_store
    gc.collect()


def load_vectorstore():
    """
    Load existing vector store.
    """

    vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME
    )

    return vector_store


# ============================================================
# CLEAR DATABASE
# ============================================================

def clear_database():
    """
    Safely clear Chroma database + uploaded docs.
    """

    try:
        print("Clearing database...")

        try:
            vector_store = Chroma(
                persist_directory=DB_DIR,
                embedding_function=embedding_model,
                collection_name=COLLECTION_NAME
            )

            vector_store.delete_collection()

            del vector_store

        except Exception:
            pass

        gc.collect()

        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR, ignore_errors=True)

        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR, ignore_errors=True)

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        os.makedirs(DB_DIR, exist_ok=True)

        print("Database cleared successfully.")

        return True

    except Exception as e:
        print("Clear DB Error:", e)
        return False


# ============================================================
# QUERY DOCUMENTS
# ============================================================

def query_documents(query, personality_mode):
    """
    Retrieve relevant chunks and generate answer.
    """

    print("\n===================================")
    print("RAG PIPELINE STARTED")
    print("===================================")

    print(f"User Query: {query}")

    vector_store = load_vectorstore()

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.7
        }
    )

    print("Searching relevant chunks...")

    docs = retriever.invoke(query)

    print(f"Retrieved {len(docs)} chunks")

    print("\n===== RETRIEVED CHUNKS =====")

    for i, doc in enumerate(docs):
        print(f"\nChunk {i+1}")
        print("SOURCE:", doc.metadata.get("source", "Unknown"))
        print("PAGE:", doc.metadata.get("page", "N/A"))
        print(doc.page_content[:300])

    print("============================")

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

    print("Sending retrieved context to LLM...")

    response = llm.invoke(final_prompt)

    del retriever
    del vector_store
    gc.collect()

    print("RAG PIPELINE COMPLETED")
    print("===================================\n")

    return response.content, docs