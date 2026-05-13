import numpy as np
from typing import Callable, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


def make_tfidf_embedder(corpus: List[str]) -> Callable[[List[str]], np.ndarray]:
    """
    Returns a pure embedding function acting as the pre-trained contrastive encoder.
    Maps text to the Semantic Manifold (X).
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit(corpus)

    def embedder(texts: List[str]) -> np.ndarray:
        vectors = vectorizer.transform(texts).toarray()
        # L2 Normalize to project onto unit hypersphere (Eq. 3 Justification)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)

    return embedder


def make_mpnet_embedder() -> Callable[[List[str]], np.ndarray]:
    """
    Returns an embedding function using the all-mpnet-base-v2 SentenceTransformer model.
    Maps text to dense semantic embeddings.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embedder(texts: List[str]) -> np.ndarray:
        return model.encode(texts, convert_to_numpy=True)

    return embedder
