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
.main {
    background-color: #0f172a;
}

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

h1, h2, h3 {
    color: white;
}

.chat-box {
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 10px;
}

.user-msg {
    background-color: #2563eb;
    color: white;
}

.bot-msg {
    background-color: #1e293b;
    color: white;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}

.stTextInput>div>div>input {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============================================================
# HEADER
# ============================================================

st.title("🧠 DocMind AI")
st.caption("Multi-Personality AI Document Assistant powered by RAG + Mistral")

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Controls")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    personality_mode = st.selectbox(
        "Choose Personality Mode",
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
# CLEAR DATABASE
# ============================================================

if clear_btn:
    with st.spinner("Clearing database..."):
        clear_database()

    st.success("Database cleared successfully.")

# ============================================================
# PROCESS DOCUMENTS
# ============================================================

if process_btn:
    if not uploaded_files:
        st.warning("Please upload at least one file.")
    else:
        with st.spinner("Processing documents..."):

            saved_paths = save_uploaded_files(uploaded_files)

            documents = load_documents(saved_paths)

            chunks = split_documents(documents)

            create_or_update_vectorstore(chunks)

        st.success("Documents processed successfully.")

# ============================================================
# QUERY INPUT
# ============================================================

query = st.chat_input("Ask a question about your documents...")

# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if "sources" in message:
            with st.expander("View Retrieved Chunks"):
                for i, doc in enumerate(message["sources"], 1):
                    st.markdown(f"### Chunk {i}")
                    st.write(doc.page_content)

# ============================================================
# HANDLE QUERY
# ============================================================

if query:
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

                with st.expander("View Retrieved Chunks"):
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