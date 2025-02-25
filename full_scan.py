# This script loads embeddings files from LAION dataset one by one,
# converts them into np array of float16 and performs full scan on them.

import os
from typing import List
import numpy as np


MAX_ID = 410


def get_img_emb_path(part: int) -> str:
    return f"data/laion/img_emb_{part}.npy"

def load_and_conver(part: int) -> np.array:
    path = get_img_emb_path(part)
    return np.load(path).astype(np.float16)


def load_all() -> List[np.ndarray]:
    # Memory efficient way to load all embeddings
    embeddings = []
    for i in range(0, MAX_ID):
        chunk = load_and_conver(i)
        print(f"Loaded chunk {i}, shape: {chunk.shape}")
        embeddings.append(chunk)
    
    return embeddings

def get_norm(embeddings: List[np.ndarray]) -> List[np.ndarray]:
    # Calculate norm for each embedding
    return [np.linalg.norm(embedding, axis=1) for embedding in embeddings]


def full_scan(embeddings: List[np.ndarray], norms: List[np.ndarray], reference: int, top: int = 50) -> np.ndarray:
    # Embeddings shape is (1_000_000, 512), we need to obtain cosine similarity

    # Get query vector
    # Output is (512,)
    query = embeddings[0][reference]

    cosine = []

    for (emb, norm) in zip(embeddings, norms):
        # Obtain dot product with each vector
        # Output is (1_000_000,)
        dot_product = np.dot(emb, query)
        
        # Obtain cosine with each vector
        # Output is (1_000_000,)
        cosine_batch = dot_product / (norm * np.linalg.norm(query))

        cosine.append(cosine_batch)

    
    cosine = np.concatenate(cosine)


    # Sort by cosine similarity
    # Output is (1_000_000,)
    sorted_indices = np.argsort(cosine)[::-1]

    # Get top
    # Output is (top,)
    top_indices = sorted_indices[:top]

    return top_indices


def main():
    # np.__config__.show()

    embeddings = load_all()

    norm = get_norm(embeddings)

    print(f"Norm shape: {norm[0].shape}, len: {len(norm)}")

    # Perform full scan for each image
    for i in range(0, 100):
        top_indices = full_scan(embeddings, norm, i)
        print(f"Top 50 images for {i}: {top_indices.tolist()}")

if __name__ == "__main__":
    main()
