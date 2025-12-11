from __future__ import annotations
from typing import List, Dict, Any
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


class VectorStore:
    """Vector store with FAISS and BM25 hybrid search."""
    
    def __init__(self, index_dir: str = "./index", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.index_dir = index_dir
        self.embedding_model_name = embedding_model
        self.encoder = None
        self.index = None
        self.bm25 = None
        self.chunks = []
        self.corpus = []
    
    def build(self, chunks: List[Any]) -> VectorStore:
        """Build FAISS and BM25 indices from document chunks."""
        import faiss
        
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Initialize encoder
        self.encoder = SentenceTransformer(self.embedding_model_name)
        
        # Store chunks
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.corpus = [t.split() for t in texts]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.corpus)
        
        # Build FAISS index
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # Save to disk
        self._save()
        
        return self
    
    def load(self) -> VectorStore:
        """Load indices from disk."""
        import faiss
        
        # Load encoder
        self.encoder = SentenceTransformer(self.embedding_model_name)
        
        # Load FAISS index
        index_path = os.path.join(self.index_dir, "faiss.index")
        self.index = faiss.read_index(index_path)
        
        # Load chunks and corpus
        with open(os.path.join(self.index_dir, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        
        with open(os.path.join(self.index_dir, "corpus.pkl"), "rb") as f:
            self.corpus = pickle.load(f)
        
        # Rebuild BM25 (it's fast enough)
        self.bm25 = BM25Okapi(self.corpus)
        
        return self
    
    def _save(self):
        """Save indices to disk."""
        import faiss
        
        # Save FAISS index
        index_path = os.path.join(self.index_dir, "faiss.index")
        faiss.write_index(self.index, index_path)
        
        # Save chunks
        with open(os.path.join(self.index_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        
        # Save corpus
        with open(os.path.join(self.index_dir, "corpus.pkl"), "wb") as f:
            pickle.dump(self.corpus, f)
    
    def search(self, query: str, top_k: int = 8, bm25_boost: float = 0.2) -> List[Dict[str, Any]]:
        """Hybrid search combining FAISS and BM25."""
        if not self.index or not self.bm25:
            raise RuntimeError("Index not loaded. Call load() first.")
        
        # Vector search
        query_embedding = self.encoder.encode([query])[0].astype('float32')
        distances, indices = self.index.search(np.array([query_embedding]), top_k * 2)
        
        # BM25 search
        query_tokens = query.split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Normalize scores
        vector_scores = 1 / (1 + distances[0])  # Convert distances to similarity scores
        bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)
        
        # Combine scores
        combined_scores = {}
        for idx, score in zip(indices[0], vector_scores):
            combined_scores[idx] = (1 - bm25_boost) * score + bm25_boost * bm25_scores_norm[idx]
        
        # Sort and return top_k
        sorted_indices = sorted(combined_scores.items(), key=lambda x: -x[1])[:top_k]
        
        results = []
        for idx, score in sorted_indices:
            chunk = self.chunks[idx]
            results.append({
                "text": chunk.text,
                "score": float(score),
                "meta": chunk.meta
            })
        
        return results
