import streamlit as st
import requests
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# ---------------- CONFIG ----------------
INDEX_PATH = "vectors/faiss_index.bin"
META_PATH = "vectors/metadata.pkl"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # must match vect.py
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3-vl:2b"   # <- use llama3
TOP_K = 5


# ---------------- LOADERS (CACHED) ----------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def load_index_and_meta():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        return None, None, None

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    texts = meta["texts"]
    metadata = meta["metadata"]
    return index, texts, metadata


# ---------------- RETRIEVER ----------------
def retrieve_context(query, top_k=TOP_K):
    model = load_embedding_model()
    index, texts, metadata = load_index_and_meta()

    if index is None:
        return [], ""

    # Embed query
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, top_k)

    results = []
    for idx in I[0]:
        chunk_text = texts[idx]
        info = metadata[idx]
        results.append((chunk_text, info))

    # Build a big context string for the LLM
    context_parts = []
    for i, (chunk_text, info) in enumerate(results):
        src = info.get("source", "unknown")
        cid = info.get("chunk_id", i)
        context_parts.append(f"[Source: {src}, Chunk {cid}]\n{chunk_text}")

    context = "\n\n".join(context_parts)
    return results, context


# ---------------- OLLAMA CALL (STRICT RAG) ----------------
def ask_ollama(question, context):
    """
    Sends question + context to Ollama /api/chat.
    Strongly instructs LLM to answer ONLY from context.
    """
    if not context.strip():
        return "❌ No context available from documents. Please upload/process files first."

    prompt = (
        "You are a Question Answering assistant for a document RAG system.\n"
        "Rules:\n"
        "1. You MUST use ONLY the information given in the context below.\n"
        "2. If the answer is not clearly present in the context, reply EXACTLY with: \"I don't know based on the provided documents.\"\n"
        "3. Do NOT use any outside knowledge, do NOT guess.\n"
        "4. If the question is vague or unrelated to the context, say you don't know.\n"
        "5. If the context contains multiple documents, combine them logically.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Now give a concise answer using ONLY the context."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "options": {
            "temperature": 0.1  # low temp to reduce hallucinations
        },
        "messages": [
            {
                "role": "system",
                "content": "You are a strict RAG assistant. You only answer from the given context and never invent facts."
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("message", {}).get("content", "").strip()
        if not answer:
            answer = "⚠️ No response content from Ollama."
        return answer
    except Exception as e:
        return f"❌ Error calling Ollama: {e}"


# ---------------- STREAMLIT UI ----------------
st.title("💬 RAG Chat (Llama3 + Your Extracted Documents)")
st.write(
    "Ask questions based on the documents you uploaded via the OCR app. "
    "Answers are generated using FAISS + SentenceTransformers + Ollama Llama3, "
    "strictly grounded on your extracted text."
)

index, texts, metadata = load_index_and_meta()
if index is None:
    st.warning(
        "❌ No vector index found.\n\n"
        "Make sure you've:\n"
        "1️⃣ Uploaded files in your OCR app (ka.py)\n"
        "2️⃣ Extracted text & saved JSON\n"
        "3️⃣ Let vect.py build the vectors (this already happens automatically from ka.py)\n"
        "Then restart this chat app."
    )
else:
    st.success(f"✅ Vector index loaded with {index.ntotal} chunks.")

show_ctx = st.sidebar.checkbox("Show retrieved context", value=False)
st.sidebar.write("Model:", OLLAMA_MODEL)
st.sidebar.write("Top-k chunks:", TOP_K)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve context
    with st.spinner("🔍 Searching in your document vectors..."):
        results, context = retrieve_context(user_input)

    if not context:
        answer = "❌ Could not load any relevant context. Make sure vectors are built and documents are not empty."
    else:
        # Optional: show retrieved chunks below the answer
        if show_ctx:
            with st.expander("🔎 Retrieved context chunks"):
                for i, (chunk_text, info) in enumerate(results):
                    st.markdown(f"**Chunk {i} — Source:** {info.get('source', 'unknown')}, ID: {info.get('chunk_id', i)}")
                    st.write(chunk_text)
                    st.markdown("---")
        # Call Ollama with context
        with st.spinner("🤖 Llama3 is thinking..."):
            answer = ask_ollama(user_input, context)
    # Show answer
    with st.chat_message("assistant"):
        st.markdown(answer)
    # Save answer in history
    st.session_state.messages.append({"role": "assistant", "content": answer})