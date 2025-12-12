
import json
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re

# ---------------- SETTINGS ----------------
JSON_DIR = "outputs"
VECTOR_DIR = "vectors"
CHUNK_SIZE = 450        # optimal for PDFs/tables
CHUNK_OVERLAP = 80      # prevents logic splits
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# --------- TEXT CLEANING ----------
def clean_text(text):
    text = re.sub(r"\s+", " ", text)     # remove extra spaces
    return text.strip()


# --------- CHUNKER (UPGRADED) ----------
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start += size - overlap
    return chunks


# --------- VECTOR BUILDER ----------
def build_vectors():
    if not os.path.exists(JSON_DIR):
        print("❌ No JSON files found! Upload files from OCR app first.")
        return

    print("🔍 Loading embedding model:", EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    all_chunks = []
    metadata = []

    for file in os.listdir(JSON_DIR):
        if file.endswith(".json"):
            path = os.path.join(JSON_DIR, file)

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            text = clean_text(data.get("extracted_text", ""))

            chunks = chunk_text(text)
            print(f"📄 {file} → {len(chunks)} chunks")

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadata.append({"source": file, "chunk_id": i})

    if not all_chunks:
        print("⚠ No text found to embed. JSON empty or unreadable.")
        return

    print("🔢 Embedding chunks...")
    vectors = model.encode(all_chunks, convert_to_numpy=True)

    os.makedirs(VECTOR_DIR, exist_ok=True)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors).astype("float32"))

    faiss.write_index(index, f"{VECTOR_DIR}/faiss_index.bin")

    with open(f"{VECTOR_DIR}/metadata.pkl", "wb") as f:
        pickle.dump({"texts": all_chunks, "metadata": metadata}, f)

    print("\n🎉 VECTORIZED SUCCESSFULLY!")
    print(f"📦 Total chunks stored: {len(all_chunks)}")
    print(f"💾 Saved to {VECTOR_DIR}/faiss_index.bin + metadata.pkl")


if __name__ == "__main__":
    build_vectors()
