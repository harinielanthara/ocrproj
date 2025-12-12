import faiss
import pickle
import numpy as np

INDEX_PATH = "vectors/faiss_index.bin"
META_PATH = "vectors/metadata.pkl"

# Load FAISS index
index = faiss.read_index(INDEX_PATH)

# Load metadata
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

texts = meta["texts"]
metadata = meta["metadata"]

print("Total chunks:", len(texts))

# Show first 3 chunks & vectors
for i in range(3):
    print("\n-------------------------------")
    print(f"Chunk {i}:")
    print("Text:", texts[i][:200], "...")  # first 200 chars
    print("Metadata:", metadata[i])

    # get vector
    vector = index.reconstruct(i)
    print("Vector length:", len(vector))
    print("Vector sample (first 10 dims):",vector)
