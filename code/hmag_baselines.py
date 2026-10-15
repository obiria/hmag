"""
The seven baseline systems compared with HMAG.

Each class re-implements the mechanism of the cited system on the retriever,
encoder, generator and prompt shared with HMAG, adapted to single-turn
question answering:

  * Static RAG (Lewis et al., NeurIPS 2020): retrieves for every query, never caches.
  * SCALM (Li et al., IWQoS 2024): online semantic-pattern clustering; pattern
    rank (by tokens saved) drives storage and eviction.
  * ContextCache (Yan et al., VLDB 2025): one cache with two-stage matching,
    a query-similarity filter followed by context-blended re-scoring.
  * CAG (Chan et al., WWW 2025): a fixed set of documents preloaded into every
    prompt, no per-query retrieval and no answer cache.
  * MinCache (Haqiq et al., FGCS 2025): one LRU cache with exact, MinHash and
    embedding matching.
  * SmartCache (Yu et al., NeurIPS 2025): reuse requires both query and
    conversational-context similarity; context-redundant entries are evicted first.
  * Adaptive-RAG (Jeong et al., NAACL 2024): routes each query to no, single-step
    or multi-step retrieval by estimated complexity; no cache.

Each benchmark question is an independent single-turn request, so the
context-aware caches see an empty conversation history.
"""

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hmag_qa_metrics import SYSTEM_PROMPT
from hmag_retrieval import Document, HierarchicalRetriever, OllamaGenerator


@dataclass
class BaselineResponse:
    response: str
    query: str
    method: str
    documents_used: List[Document]
    cache_hit: bool
    used_retrieval: bool
    latency_breakdown: Dict[str, float]
    total_latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaselineSystem(ABC):
    name: str = "baseline"

    def __init__(self, retriever: HierarchicalRetriever, generator: OllamaGenerator,
                 top_k: int = 10):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def _retrieve_from_corpus(self, query: str, k: Optional[int] = None) -> List[Document]:
        result = asyncio.run(self.retriever.retrieve_async(query, k=k or self.top_k, strategy="dense"))
        return result.documents

    def _generate(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        return self.generator.generate(prompt=query, context=documents, system_prompt=SYSTEM_PROMPT)

    def _finalize(self, response: Dict[str, Any], query: str, documents: List[Document],
                  cache_hit: bool, used_retrieval: bool, latency_breakdown: Dict[str, float],
                  total_latency_ms: float, metadata: Optional[Dict[str, Any]] = None) -> BaselineResponse:
        """Builds the response record, adding the generation timings and token counts
        when the generator was called."""
        latency_breakdown = dict(latency_breakdown)
        metadata = dict(metadata or {})
        if 'ttft_ms' in response:
            latency_breakdown['time_to_first_token'] = response['ttft_ms']
            latency_breakdown['decoding'] = response.get('decoding_ms', 0.0)
            metadata['prompt_tokens'] = response.get('prompt_tokens', 0)
            metadata['generated_tokens'] = response.get('generated_tokens', 0)
        metadata['generation_error'] = bool(response.get('error'))
        return BaselineResponse(
            response=response['response'], query=query, method=self.name,
            documents_used=documents, cache_hit=cache_hit, used_retrieval=used_retrieval,
            latency_breakdown=latency_breakdown, total_latency_ms=total_latency_ms, metadata=metadata,
        )

    @abstractmethod
    def process_query(self, query: str) -> BaselineResponse:
        ...


class _FlatSemanticCache:
    """Embedding cache with cosine-similarity lookup and LRU or LFU eviction
    (semantic stage of MinCache)."""

    def __init__(self, encoder, capacity: int, similarity_threshold: float, policy: str):
        self.encoder = encoder
        self.capacity = capacity
        self.threshold = similarity_threshold
        self.policy = policy  # "lru" | "lfu"
        self.entries: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.embedding_dim = encoder.get_sentence_embedding_dimension()

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return (v / n).astype('float32') if n > 0 else v.astype('float32')

    def lookup(self, query: str) -> Tuple[Optional[Document], bool]:
        if not self.entries:
            return None, False
        q_emb = self._normalize(self.encoder.encode(query, convert_to_numpy=True))
        keys = list(self.entries.keys())
        embeddings = np.stack([self.entries[k]['embedding'] for k in keys])
        sims = embeddings @ q_emb
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= self.threshold:
            key = keys[best_idx]
            self.entries.move_to_end(key)
            self.access_counts[key] += 1
            entry = self.entries[key]
            return Document(doc_id=key, content=entry['content'],
                             relevance_score=float(sims[best_idx]), source_type="cache"), True
        return None, False

    def insert(self, query: str, content: str):
        """Stores the response under the embedding of its query."""
        key = hashlib.sha256(query.encode()).hexdigest()
        if key in self.entries:
            self.entries.move_to_end(key)
            return
        embedding = self._normalize(self.encoder.encode(query, convert_to_numpy=True))
        self.entries[key] = {'content': content, 'embedding': embedding}
        self.access_counts[key] = 1

        if len(self.entries) > self.capacity:
            if self.policy == "lfu":
                victim = min(self.entries.keys(), key=lambda k: self.access_counts[k])
            else:  # lru: least recently used entry first
                victim = next(iter(self.entries))
            del self.entries[victim]
            del self.access_counts[victim]


class SCALMBaseline(BaselineSystem):
    """SCALM (Li et al., IWQoS 2024): queries are clustered online into semantic
    patterns (nearest-centroid assignment to at most `num_patterns` centroids, in
    place of the batch clustering of the original). Patterns are ranked by the tokens
    their hits have saved; once the cache is full, only queries of high- and mid-rank
    patterns are stored. Entries receive eviction priority 3, 2 or 1 by rank, plus 1
    per hit, and the entry of lowest priority (oldest first on ties) is evicted."""
    name = "SCALM"

    def __init__(self, retriever, generator, capacity: int = 1000,
                 similarity_threshold: float = 0.85, num_patterns: int = 20, **kwargs):
        super().__init__(retriever, generator, **kwargs)
        self.capacity = capacity
        self.threshold = similarity_threshold
        self.num_patterns = num_patterns
        self.encoder = retriever.encoder
        # key -> {embedding, content, pattern, priority, insert_order}
        self.entries: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._insert_counter = 0
        self.pattern_centroids: List[np.ndarray] = []
        self.pattern_counts: List[int] = []          # queries assigned to each pattern
        self.pattern_tokens_saved: List[int] = []     # cumulative generated-tokens saved by hits per pattern

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return (v / n).astype('float32') if n > 0 else v.astype('float32')

    def _assign_pattern(self, embedding: np.ndarray) -> int:
        if self.pattern_centroids:
            sims = [float(np.dot(embedding, c)) for c in self.pattern_centroids]
            best = int(np.argmax(sims))
            if sims[best] >= self.threshold:
                n = self.pattern_counts[best]
                self.pattern_centroids[best] = self._normalize(
                    (self.pattern_centroids[best] * n + embedding) / (n + 1))
                self.pattern_counts[best] += 1
                return best
        if len(self.pattern_centroids) < self.num_patterns:
            self.pattern_centroids.append(embedding)
            self.pattern_counts.append(1)
            self.pattern_tokens_saved.append(0)
            return len(self.pattern_centroids) - 1
        return int(np.argmax([float(np.dot(embedding, c)) for c in self.pattern_centroids]))

    def _pattern_rank(self, pattern_id: int) -> str:
        """High, mid or low: top, middle or bottom third of the patterns by tokens saved."""
        if len(self.pattern_tokens_saved) <= 1:
            return "high"
        order = sorted(range(len(self.pattern_tokens_saved)),
                        key=lambda i: self.pattern_tokens_saved[i], reverse=True)
        rank_idx = order.index(pattern_id)
        third = max(1, len(order) // 3)
        if rank_idx < third:
            return "high"
        if rank_idx < 2 * third:
            return "mid"
        return "low"

    def _lookup(self, query: str) -> Tuple[Optional[Dict], bool]:
        if not self.entries:
            return None, False
        q_emb = self._normalize(self.encoder.encode(query, convert_to_numpy=True))
        keys = list(self.entries.keys())
        embeddings = np.stack([self.entries[k]['embedding'] for k in keys])
        sims = embeddings @ q_emb
        best_idx = int(np.argmax(sims))
        if sims[best_idx] < self.threshold:
            return None, False
        key = keys[best_idx]
        entry = self.entries[key]
        entry['priority'] += 1
        self.pattern_tokens_saved[entry['pattern']] += entry.get('generated_tokens', 0)
        self.entries.move_to_end(key)
        return entry, True

    def _evict_if_needed(self):
        while len(self.entries) > self.capacity:
            # lowest priority first, oldest first on ties
            victim_key = min(
                self.entries.keys(),
                key=lambda k: (self.entries[k]['priority'], self.entries[k]['insert_order']),
            )
            del self.entries[victim_key]

    def process_query(self, query: str) -> BaselineResponse:
        start = time.perf_counter()
        cache_start = time.perf_counter()
        entry, hit = self._lookup(query)
        cache_latency = (time.perf_counter() - cache_start) * 1000

        retrieval_latency = gen_latency = 0.0
        used_retrieval = False
        documents: List[Document] = []
        if hit:
            documents = [Document(doc_id=hashlib.sha256(query.encode()).hexdigest(),
                                   content=entry['content'], relevance_score=1.0, source_type="cache")]
            response = {'response': entry['content']}
        else:
            r_start = time.perf_counter()
            documents = self._retrieve_from_corpus(query)
            retrieval_latency = (time.perf_counter() - r_start) * 1000
            used_retrieval = True

            gen_start = time.perf_counter()
            response = self._generate(query, documents)
            gen_latency = (time.perf_counter() - gen_start) * 1000

            q_emb = self._normalize(self.encoder.encode(query, convert_to_numpy=True))
            pattern_id = self._assign_pattern(q_emb)
            rank = self._pattern_rank(pattern_id)
            # Below capacity every answer is stored; once full, only high- and mid-rank
            # patterns. Failed generations are not stored.
            is_cold = len(self.entries) < self.capacity
            if not response.get('error') and (is_cold or rank in ("mid", "high")):
                self._insert_counter += 1
                self.entries[hashlib.sha256(query.encode()).hexdigest()] = {
                    'content': response['response'], 'embedding': q_emb, 'pattern': pattern_id,
                    'priority': {"high": 3, "mid": 2, "low": 1}[rank],
                    'insert_order': self._insert_counter,
                    'generated_tokens': response.get('generated_tokens', 0),
                }
                self._evict_if_needed()

        total = (time.perf_counter() - start) * 1000
        return self._finalize(
            response, query, documents, cache_hit=hit, used_retrieval=used_retrieval,
            latency_breakdown={'cache': cache_latency, 'retrieval': retrieval_latency, 'generation': gen_latency},
            total_latency_ms=total,
        )


class StaticRAGBaseline(BaselineSystem):
    """Static RAG (Lewis et al., NeurIPS 2020): retrieves for every query, no cache."""
    name = "Static RAG"

    def process_query(self, query: str) -> BaselineResponse:
        start = time.perf_counter()
        r_start = time.perf_counter()
        documents = self._retrieve_from_corpus(query)
        retrieval_latency = (time.perf_counter() - r_start) * 1000

        gen_start = time.perf_counter()
        response = self._generate(query, documents)
        gen_latency = (time.perf_counter() - gen_start) * 1000

        total = (time.perf_counter() - start) * 1000
        return self._finalize(
            response, query, documents, cache_hit=False, used_retrieval=True,
            latency_breakdown={'retrieval': retrieval_latency, 'generation': gen_latency},
            total_latency_ms=total,
        )


class AdaptiveRAGBaseline(BaselineSystem):
    """Adaptive-RAG (Jeong et al., NAACL 2024): routes each query to no retrieval,
    single-step retrieval or two-round retrieval by estimated complexity. Complexity
    is estimated heuristically (query length, comparative markers, digits and
    capitalized words) instead of with the trained classifier of the original.
    No cache."""
    name = "Adaptive-RAG"

    _MULTI_HOP_MARKERS = ("compare", "compared", "difference between", " and ",
                           " versus", " vs ", "both", "either", "relationship between")

    @classmethod
    def _classify(cls, query: str) -> str:
        tokens = query.strip().split()
        if not tokens:
            return "simple"
        lower = query.lower()
        has_digit = any(ch.isdigit() for ch in query)
        has_entity = any(t[:1].isupper() for t in tokens[1:] if t[:1].isalpha())
        is_multi_hop = len(tokens) > 15 or any(m in lower for m in cls._MULTI_HOP_MARKERS)
        if is_multi_hop:
            return "complex"
        if has_digit or has_entity:
            return "moderate"
        return "simple"

    def process_query(self, query: str) -> BaselineResponse:
        start = time.perf_counter()
        complexity = self._classify(query)
        retrieval_latency = 0.0
        used_retrieval = complexity != "simple"
        documents: List[Document] = []

        if complexity == "moderate":
            r_start = time.perf_counter()
            documents = self._retrieve_from_corpus(query)
            retrieval_latency = (time.perf_counter() - r_start) * 1000
        elif complexity == "complex":
            # second round with the query extended by the top passage
            r_start = time.perf_counter()
            first_k = max(1, self.top_k // 2)
            first = self._retrieve_from_corpus(query, k=first_k)
            documents = list(first)
            if first:
                follow_up_query = f"{query} {first[0].content[:200]}"
                second = self._retrieve_from_corpus(follow_up_query, k=first_k)
                seen = {d.doc_id for d in documents}
                documents += [d for d in second if d.doc_id not in seen]
            retrieval_latency = (time.perf_counter() - r_start) * 1000

        gen_start = time.perf_counter()
        response = self._generate(query, documents)
        gen_latency = (time.perf_counter() - gen_start) * 1000

        total = (time.perf_counter() - start) * 1000
        return self._finalize(
            response, query, documents, cache_hit=False, used_retrieval=used_retrieval,
            latency_breakdown={'retrieval': retrieval_latency, 'generation': gen_latency},
            total_latency_ms=total, metadata={'complexity': complexity},
        )


class ContextCacheBaseline(BaselineSystem):
    """ContextCache (Yan et al., VLDB 2025): two-stage matching. Stage 1 shortlists
    entries by query similarity; stage 2 re-scores them with a representation that
    blends the query with a recency-weighted history of the session's earlier queries
    (in place of the attention-based context model of the original). A hit must pass
    both stages; eviction is LRU. Each benchmark question is its own single-turn
    session, so the history is empty."""
    name = "ContextCache"

    def __init__(self, retriever, generator, capacity: int = 1000,
                 candidate_threshold: float = 0.75, context_threshold: float = 0.85,
                 context_window: int = 3, context_decay: float = 0.5, single_turn: bool = True, **kwargs):
        super().__init__(retriever, generator, **kwargs)
        self.capacity = capacity
        self.candidate_threshold = candidate_threshold
        self.context_threshold = context_threshold
        self.context_decay = context_decay
        self.single_turn = single_turn
        self.encoder = retriever.encoder
        self.entries: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()  # key -> {embedding, context, content}
        self.recent_queries: deque = deque(maxlen=context_window)

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return (v / n).astype('float32') if n > 0 else v.astype('float32')

    def _context_repr(self, current_emb: np.ndarray) -> np.ndarray:
        vecs = [current_emb]
        weights = [1.0]
        for i, e in enumerate(reversed(self.recent_queries)):
            vecs.append(e)
            weights.append(self.context_decay ** (i + 1))
        weights = np.array(weights) / sum(weights)
        blended = sum(w * v for w, v in zip(weights, vecs))
        return self._normalize(blended)

    def _lookup(self, query: str) -> Tuple[Optional[Dict], bool]:
        if not self.entries:
            return None, False
        q_emb = self._normalize(self.encoder.encode(query, convert_to_numpy=True))

        # Stage 1: coarse candidate filter on the query embedding alone.
        candidates = [(k, e) for k, e in self.entries.items()
                      if float(np.dot(q_emb, e['embedding'])) >= self.candidate_threshold]
        if not candidates:
            return None, False

        # Stage 2: contextual re-scoring.
        ctx_repr = self._context_repr(q_emb)
        best_key, best_score = None, -1.0
        for key, entry in candidates:
            score = float(np.dot(ctx_repr, entry['context']))
            if score > best_score:
                best_key, best_score = key, score
        if best_score < self.context_threshold:
            return None, False
        self.entries.move_to_end(best_key)
        return self.entries[best_key], True

    def _evict_if_needed(self):
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)

    def process_query(self, query: str) -> BaselineResponse:
        start = time.perf_counter()
        cache_start = time.perf_counter()
        entry, hit = self._lookup(query)
        cache_latency = (time.perf_counter() - cache_start) * 1000

        q_emb = self._normalize(self.encoder.encode(query, convert_to_numpy=True))
        ctx_repr = self._context_repr(q_emb)  # context before this query joins the history
        if not self.single_turn:
            self.recent_queries.append(q_emb)

        retrieval_latency = gen_latency = 0.0
        used_retrieval = False
        documents: List[Document] = []
        if hit:
            documents = [Document(doc_id=hashlib.sha256(query.encode()).hexdigest(),
                                   content=entry['content'], relevance_score=1.0, source_type="cache")]
            response = {'response': entry['content']}
        else:
            r_start = time.perf_counter()
            documents = self._retrieve_from_corpus(query)
            retrieval_latency = (time.perf_counter() - r_start) * 1000
            used_retrieval = True

            gen_start = time.perf_counter()
            response = self._generate(query, documents)
            gen_latency = (time.perf_counter() - gen_start) * 1000

            if not response.get('error'):
                key = hashlib.sha256(query.encode()).hexdigest()
                self.entries[key] = {'content': response['response'], 'embedding': q_emb, 'context': ctx_repr}
                self._evict_if_needed()

        total = (time.perf_counter() - start) * 1000
        return self._finalize(
            response, query, documents, cache_hit=hit, used_retrieval=used_retrieval,
            latency_breakdown={'cache': cache_latency, 'retrieval': retrieval_latency, 'generation': gen_latency},
            total_latency_ms=total,
        )


class CAGBaseline(BaselineSystem):
    """CAG (Chan et al., WWW 2025): a fixed set of corpus documents is placed in every
    prompt; no per-query retrieval and no answer cache. Documents are preloaded in
    corpus order while their estimated size (at most 1,000 characters each, about 3.5
    characters per token) stays within `context_token_budget`, which leaves room in
    the 4,096-token window for the instruction, the question and the answer. The model
    server reuses the key-value states of this unchanged prompt prefix."""
    name = "CAG"

    def __init__(self, retriever, generator, static_context_docs: int = 50,
                 context_token_budget: int = 3072, chars_per_token: float = 3.5, **kwargs):
        super().__init__(retriever, generator, **kwargs)
        self.static_context_docs = static_context_docs
        self.context_token_budget = context_token_budget
        self.chars_per_token = chars_per_token
        self._preloaded: Optional[List[Document]] = None

    def preload(self, documents: List[Document]):
        chosen, used = [], 0.0
        for doc in documents[:self.static_context_docs]:
            cost = (min(len(doc.content), 1000) + 40) / self.chars_per_token
            if chosen and used + cost > self.context_token_budget:
                break
            chosen.append(doc)
            used += cost
        self._preloaded = chosen

    def process_query(self, query: str) -> BaselineResponse:
        start = time.perf_counter()
        documents = self._preloaded or []

        gen_start = time.perf_counter()
        response = self._generate(query, documents)
        gen_latency = (time.perf_counter() - gen_start) * 1000

        total = (time.perf_counter() - start) * 1000
        return self._finalize(
            response, query, documents, cache_hit=False, used_retrieval=False,
            latency_breakdown={'generation': gen_latency},
            total_latency_ms=total,
        )


class _MinHashIndex:
    """MinHash index over word 3-shingles (universal hashing) for Jaccard
    resemblance matching (second stage of MinCache)."""

    def __init__(self, num_hashes: int = 64, shingle_size: int = 3, seed: int = 13):
        self.shingle_size = shingle_size
        self._prime = (1 << 61) - 1
        rng = np.random.default_rng(seed)
        self._a = rng.integers(1, self._prime, size=num_hashes, dtype=np.int64)
        self._b = rng.integers(0, self._prime, size=num_hashes, dtype=np.int64)
        self.signatures: Dict[str, np.ndarray] = {}

    def _shingles(self, text: str) -> set:
        tokens = text.lower().split()
        if len(tokens) < self.shingle_size:
            return {" ".join(tokens)} if tokens else set()
        return {" ".join(tokens[i:i + self.shingle_size])
                for i in range(len(tokens) - self.shingle_size + 1)}

    def _signature(self, text: str) -> np.ndarray:
        shingles = self._shingles(text)
        if not shingles:
            return np.full(len(self._a), self._prime, dtype=np.int64)
        hashed = np.array(
            [int(hashlib.md5(s.encode()).hexdigest(), 16) % self._prime for s in shingles],
            dtype=np.int64,
        )
        vals = (np.outer(self._a, hashed) + self._b[:, None]) % self._prime
        return vals.min(axis=1)

    def insert(self, key: str, text: str):
        self.signatures[key] = self._signature(text)

    def remove(self, key: str):
        self.signatures.pop(key, None)

    def best_match(self, text: str) -> Tuple[Optional[str], float]:
        if not self.signatures:
            return None, 0.0
        sig = self._signature(text)
        best_key, best_score = None, 0.0
        for key, other_sig in self.signatures.items():
            score = float(np.mean(sig == other_sig))  # estimate of the Jaccard similarity
            if score > best_score:
                best_key, best_score = key, score
        return best_key, best_score


class MinCacheBaseline(BaselineSystem):
    """MinCache (Haqiq et al., FGCS 2025): one LRU cache with a three-stage lookup:
    exact match, MinHash resemblance match and embedding similarity match."""
    name = "MinCache"

    def __init__(self, retriever, generator, capacity: int = 1000,
                 resemblance_threshold: float = 0.5, semantic_threshold: float = 0.85, **kwargs):
        super().__init__(retriever, generator, **kwargs)
        self.capacity = capacity
        self.resemblance_threshold = resemblance_threshold
        self.exact: "OrderedDict[str, str]" = OrderedDict()  # normalized query -> response
        self.minhash = _MinHashIndex()
        self.semantic = _FlatSemanticCache(retriever.encoder, capacity, semantic_threshold, policy="lru")

    @staticmethod
    def _norm(query: str) -> str:
        return " ".join(query.lower().split())

    def _evict_if_needed(self):
        while len(self.exact) > self.capacity:
            victim_key, _ = self.exact.popitem(last=False)
            self.minhash.remove(victim_key)

    def process_query(self, query: str) -> BaselineResponse:
        start = time.perf_counter()
        cache_start = time.perf_counter()
        norm_q = self._norm(query)
        documents: List[Document] = []
        response_text: Optional[str] = None
        match_tier: Optional[str] = None

        # Stage 1: exact match.
        if norm_q in self.exact:
            self.exact.move_to_end(norm_q)
            response_text, match_tier = self.exact[norm_q], "exact"
        else:
            # Stage 2: MinHash/Jaccard resemblance match.
            match_key, score = self.minhash.best_match(norm_q)
            if match_key is not None and score >= self.resemblance_threshold and match_key in self.exact:
                self.exact.move_to_end(match_key)
                response_text, match_tier = self.exact[match_key], "resemblance"
            else:
                # Stage 3: semantic cosine-similarity match.
                cached_doc, sem_hit = self.semantic.lookup(query)
                if sem_hit:
                    response_text, match_tier = cached_doc.content, "semantic"
                    documents = [cached_doc]
        cache_latency = (time.perf_counter() - cache_start) * 1000

        hit = response_text is not None
        retrieval_latency = gen_latency = 0.0
        used_retrieval = False
        if hit:
            response = {'response': response_text}
        else:
            r_start = time.perf_counter()
            documents = self._retrieve_from_corpus(query)
            retrieval_latency = (time.perf_counter() - r_start) * 1000
            used_retrieval = True

            gen_start = time.perf_counter()
            response = self._generate(query, documents)
            gen_latency = (time.perf_counter() - gen_start) * 1000

            if not response.get('error'):
                self.exact[norm_q] = response['response']
                self.minhash.insert(norm_q, norm_q)
                self.semantic.insert(query, response['response'])
                self._evict_if_needed()

        total = (time.perf_counter() - start) * 1000
        return self._finalize(
            response, query, documents, cache_hit=hit, used_retrieval=used_retrieval,
            latency_breakdown={'cache': cache_latency, 'retrieval': retrieval_latency, 'generation': gen_latency},
            total_latency_ms=total, metadata={'match_tier': match_tier},
        )


class SmartCacheBaseline(BaselineSystem):
    """SmartCache (Yu et al., NeurIPS 2025): reuse requires query similarity and
    conversational-context similarity (context: mean embedding of the session's last
    `context_window` queries). Eviction removes the most context-redundant entry among
    the oldest candidates. With single-turn sessions only query similarity applies."""
    name = "SmartCache"

    def __init__(self, retriever, generator, capacity: int = 1000,
                 query_threshold: float = 0.85, context_threshold: float = 0.6,
                 context_window: int = 3, single_turn: bool = True, **kwargs):
        super().__init__(retriever, generator, **kwargs)
        self.capacity = capacity
        self.query_threshold = query_threshold
        self.context_threshold = context_threshold
        self.single_turn = single_turn
        self.encoder = retriever.encoder
        self.entries: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.recent_queries: deque = deque(maxlen=context_window)

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return (v / n).astype('float32') if n > 0 else v.astype('float32')

    def _context_signature(self) -> Optional[np.ndarray]:
        if not self.recent_queries:
            return None
        return self._normalize(np.mean(np.stack(list(self.recent_queries)), axis=0))

    def _lookup(self, query: str, context_sig: Optional[np.ndarray]) -> Tuple[Optional[Dict], bool]:
        if not self.entries:
            return None, False
        q_emb = self._normalize(self.encoder.encode(query, convert_to_numpy=True))
        best_key, best_score = None, -1.0
        for key, entry in self.entries.items():
            q_sim = float(np.dot(q_emb, entry['embedding']))
            if q_sim < self.query_threshold:
                continue
            if context_sig is not None and entry['context'] is not None:
                c_sim = float(np.dot(context_sig, entry['context']))
                if c_sim < self.context_threshold:
                    continue
            if q_sim > best_score:
                best_key, best_score = key, q_sim
        if best_key is None:
            return None, False
        self.entries.move_to_end(best_key)
        return self.entries[best_key], True

    def _evict_if_needed(self):
        while len(self.entries) > self.capacity:
            overflow = len(self.entries) - self.capacity
            candidates = list(self.entries.items())[:max(overflow * 4, 20)]  # oldest entries
            best_key, best_redundancy = candidates[0][0], -1
            for key, entry in candidates:
                if entry['context'] is None:
                    redundancy = 0
                else:
                    redundancy = sum(
                        1 for k2, e2 in candidates
                        if k2 != key and e2['context'] is not None
                        and float(np.dot(entry['context'], e2['context'])) >= self.context_threshold
                    )
                if redundancy > best_redundancy:
                    best_key, best_redundancy = key, redundancy
            del self.entries[best_key]

    def process_query(self, query: str) -> BaselineResponse:
        start = time.perf_counter()
        cache_start = time.perf_counter()
        context_sig = self._context_signature()
        entry, hit = self._lookup(query, context_sig)
        cache_latency = (time.perf_counter() - cache_start) * 1000

        q_emb = self._normalize(self.encoder.encode(query, convert_to_numpy=True))
        if not self.single_turn:
            self.recent_queries.append(q_emb)  # context of the next query

        retrieval_latency = gen_latency = 0.0
        used_retrieval = False
        documents: List[Document] = []
        if hit:
            documents = [Document(doc_id=hashlib.sha256(query.encode()).hexdigest(),
                                   content=entry['content'], relevance_score=1.0, source_type="cache")]
            response = {'response': entry['content']}
        else:
            r_start = time.perf_counter()
            documents = self._retrieve_from_corpus(query)
            retrieval_latency = (time.perf_counter() - r_start) * 1000
            used_retrieval = True

            gen_start = time.perf_counter()
            response = self._generate(query, documents)
            gen_latency = (time.perf_counter() - gen_start) * 1000

            if not response.get('error'):
                key = hashlib.sha256(query.encode()).hexdigest()
                self.entries[key] = {'content': response['response'], 'embedding': q_emb, 'context': context_sig}
                self._evict_if_needed()

        total = (time.perf_counter() - start) * 1000
        return self._finalize(
            response, query, documents, cache_hit=hit, used_retrieval=used_retrieval,
            latency_breakdown={'cache': cache_latency, 'retrieval': retrieval_latency, 'generation': gen_latency},
            total_latency_ms=total,
        )


def build_all_baselines(retriever: HierarchicalRetriever, generator: OllamaGenerator,
                         top_k: int = 10, capacity: Optional[int] = None) -> Dict[str, BaselineSystem]:
    """The seven baselines of Table 2. `capacity` sets the number of answers of every
    answer cache (default 1,000)."""
    cap = {} if capacity is None else {"capacity": capacity}
    return {
        "Static RAG": StaticRAGBaseline(retriever, generator, top_k=top_k),
        "SCALM": SCALMBaseline(retriever, generator, top_k=top_k, **cap),
        "ContextCache": ContextCacheBaseline(retriever, generator, top_k=top_k, **cap),
        "CAG": CAGBaseline(retriever, generator, top_k=top_k),
        "MinCache": MinCacheBaseline(retriever, generator, top_k=top_k, **cap),
        "SmartCache": SmartCacheBaseline(retriever, generator, top_k=top_k, **cap),
        "Adaptive-RAG": AdaptiveRAGBaseline(retriever, generator, top_k=top_k),
    }
