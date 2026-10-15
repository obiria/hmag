"""Dense passage retrieval (FAISS exact inner product over normalized embeddings) and
the Ollama generator shared by HMAG and all baselines."""

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import ollama
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class Document:
    doc_id: str
    content: str
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    relevance_score: float = 0.0
    source_type: str = "unknown"  # "retrieval" or "cache"


@dataclass
class RetrievalResult:
    documents: List[Document]
    scores: List[float]
    latency_ms: float
    strategy_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class HierarchicalRetriever:
    """Corpus retriever. Documents are embedded once; a query is answered with the k
    passages of highest cosine similarity."""

    def __init__(self,
                 model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 use_gpu: bool = torch.cuda.is_available(),
                 query_embed_cache: Optional["OrderedDict[str, np.ndarray]"] = None):
        """`query_embed_cache` memoizes raw query embeddings by exact text (LRU, 2,000
        queries); HMAG shares it with its caches so that a query is encoded once."""
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name, device='cuda' if use_gpu else 'cpu')
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self._query_embed_cache: "OrderedDict[str, np.ndarray]" = (
            query_embed_cache if query_embed_cache is not None else OrderedDict())
        self._query_embed_cache_max = 2000
        self.dense_index = faiss.IndexFlatIP(self.embedding_dim)
        self.documents: Dict[str, Document] = {}

    def encode_query_cached(self, query: str) -> np.ndarray:
        """Raw (unnormalized) query embedding."""
        cached = self._query_embed_cache.get(query)
        if cached is not None:
            self._query_embed_cache.move_to_end(query)
            return cached
        embedding = self.encoder.encode(query, convert_to_numpy=True)
        self._query_embed_cache[query] = embedding
        if len(self._query_embed_cache) > self._query_embed_cache_max:
            self._query_embed_cache.popitem(last=False)
        return embedding

    def add_documents(self, documents: List[Document], batch_size: int = 32):
        """Embeds and indexes documents; the index position of a document equals its
        insertion order in `self.documents`."""
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            embeddings = self.encoder.encode([doc.content for doc in batch], convert_to_numpy=True,
                                             batch_size=batch_size, show_progress_bar=False)
            for doc, embedding in zip(batch, embeddings):
                doc.embedding = embedding
                self.documents[doc.doc_id] = doc
            embeddings = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings)
            self.dense_index.add(embeddings)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # return the encoder's batch buffers to the GPU

    async def retrieve_async(self, query: str, k: int = 10, strategy: str = "dense") -> RetrievalResult:
        """Returns the top-k documents for `query`; only dense retrieval is implemented."""
        if strategy != "dense":
            raise ValueError(f"Unsupported retrieval strategy: {strategy}")
        start = time.perf_counter()
        try:
            query_embedding = self.encode_query_cached(query).astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            scores, indices = self.dense_index.search(query_embedding, k)
            doc_list = list(self.documents.values())
            documents, result_scores = [], []
            for idx, score in zip(indices[0], scores[0]):
                if 0 <= idx < len(doc_list):
                    doc = doc_list[idx]
                    doc.relevance_score = float(score)
                    documents.append(doc)
                    result_scores.append(float(score))
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            documents, result_scores = [], []
        return RetrievalResult(documents=documents, scores=result_scores,
                               latency_ms=(time.perf_counter() - start) * 1000, strategy_used="dense")


class OllamaGenerator:
    """Generator served by Ollama. The prompt consists of the instruction, the context
    documents (at most 1,000 characters each) and the question."""

    def __init__(self,
                 model_name: str = "llama3.2:latest",
                 host: str = "http://localhost:11434",
                 temperature: float = 0.7,
                 max_tokens: int = 1024,
                 max_context_docs: Optional[int] = None,
                 num_ctx: int = 4096):
        self.model_name = model_name
        self.host = host
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_context_docs = max_context_docs  # None: use all documents passed in
        self.num_ctx = num_ctx
        self.client = ollama.Client(host=host)

    def generate(self, prompt: str, context: Optional[List[Document]] = None,
                 system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generates an answer. Durations reported by Ollama are converted to milliseconds:
        time to first token = model load + prompt evaluation; decoding = evaluation."""
        context_text = ""
        if context:
            docs = context[:self.max_context_docs] if self.max_context_docs else context
            context_text = "\n\n".join(
                f"Document {i + 1} (Relevance: {doc.relevance_score:.2f}):\n{doc.content[:1000]}"
                for i, doc in enumerate(docs))

        full_prompt = ""
        if system_prompt:
            full_prompt += f"System: {system_prompt}\n\n"
        if context_text:
            full_prompt += f"Context:\n{context_text}\n\n"
        full_prompt += f"Query: {prompt}\n\nResponse:"

        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=full_prompt,
                options={'temperature': self.temperature, 'num_predict': self.max_tokens,
                         'num_ctx': self.num_ctx, 'top_k': 50, 'top_p': 0.95},
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {'response': f"Error generating response: {e}", 'error': str(e)}

        load_ns = response.get('load_duration', 0) or 0
        prefill_ns = response.get('prompt_eval_duration', 0) or 0
        decode_ns = response.get('eval_duration', 0) or 0
        return {
            'response': response.get('response', ''),
            'context': context_text,
            'total_duration': response.get('total_duration', 0),
            'ttft_ms': (load_ns + prefill_ns) / 1e6,
            'decoding_ms': decode_ns / 1e6,
            'prompt_tokens': response.get('prompt_eval_count', 0),
            'generated_tokens': response.get('eval_count', 0),
        }
