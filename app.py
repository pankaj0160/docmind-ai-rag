# app.py

import streamlit as st

from rag_pipeline import (
    save_uploaded_files,
    load_documents,
    split_documents,
    create_or_update_vectorstore,
    query_documents,
    clear_database
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

h1, h2, h3 {
    color: white;
}

.stChatMessage {
    border-radius: 15px;
    padding: 10px;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    font-weight: bold;
    border: none;
}

.stTextInput > div > div > input {
    border-radius: 12px;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False

if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []

# ============================================================
# RESET APP FUNCTION
# ============================================================

def reset_app():
    st.session_state.chat_history = []
    st.session_state.docs_processed = False
    st.session_state.uploaded_file_names = []

# ============================================================
# HEADER
# ============================================================

st.title("🧠 DocMind AI")
st.caption("AI-Powered Multi-Personality Document Assistant using RAG + Mistral")

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Controls")

    uploaded_files = st.file_uploader(
        "📂 Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    personality_mode = st.selectbox(
        "🎭 Choose Personality",
        [
            "Professional",
            "Fun",
            "Nerd",
            "Sad",
            "Classic Indian"
        ]
    )

    st.markdown("---")

    process_btn = st.button("📄 Process Documents")
    clear_btn = st.button("🗑 Clear Database")

# ============================================================
# FILE REMOVAL DETECTION
# ============================================================

current_file_names = []

if uploaded_files:
    current_file_names = [file.name for file in uploaded_files]

if (
    st.session_state.uploaded_file_names
    and current_file_names != st.session_state.uploaded_file_names
):
    clear_database()
    reset_app()
    st.warning("Uploaded files changed. Previous session cleared.")
    st.rerun()

# ============================================================
# CLEAR DATABASE
# ============================================================

if clear_btn:
    with st.spinner("Clearing database..."):
        success = clear_database()

    if success:
        reset_app()
        st.success("Database cleared successfully.")
        st.rerun()
    else:
        st.error("Failed to clear database.")

# ============================================================
# PROCESS DOCUMENTS
# ============================================================

if process_btn:
    if not uploaded_files:
        st.warning("Please upload at least one file.")
    else:
        with st.spinner("Processing documents..."):

            # Replace old documents completely
            clear_database()

            saved_paths = save_uploaded_files(uploaded_files)

            documents = load_documents(saved_paths)

            chunks = split_documents(documents)

            create_or_update_vectorstore(chunks)

            st.session_state.docs_processed = True
            st.session_state.uploaded_file_names = current_file_names
            st.session_state.chat_history = []

        st.success("Documents processed successfully.")
        st.rerun()

# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.docs_processed:
    st.info("📂 Upload documents and click 'Process Documents' to start chatting.")

# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if "sources" in message:
            with st.expander("📄 View Retrieved Chunks"):
                for i, doc in enumerate(message["sources"], 1):
                    st.markdown(f"### Chunk {i}")
                    st.write(doc.page_content)

# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input("Ask a question about your documents...")

# ============================================================
# HANDLE QUERY
# ============================================================

if query:

    if not st.session_state.docs_processed:
        st.warning("Please upload and process documents first.")
        st.stop()

    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                answer, docs = query_documents(
                    query,
                    personality_mode
                )

                st.markdown(answer)

                with st.expander("📄 View Retrieved Chunks"):
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"### Chunk {i}")
                        st.write(doc.page_content)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": docs
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")