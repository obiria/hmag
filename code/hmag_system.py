"""HMAG serving pipeline: hierarchical answer cache, retrieval-strategy controller,
consistency verifier and generator."""

import asyncio
import hashlib
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from hmag_core import (
    FixedStrategyController,
    HierarchicalCache,
    LinearThompsonSamplingController,
    PolicyGradientController,
    QueryContext,
    RetrievalStrategy,
)
from hmag_qa_metrics import SYSTEM_PROMPT
from hmag_retrieval import Document, HierarchicalRetriever, OllamaGenerator
from hmag_verifier import AdvancedVerifier, ConsistencyResult

logger = logging.getLogger(__name__)

# Controller state x_t = [phi(q_t), h_cache, h_temporal] (Algorithm 2): h_cache holds the
# hit rate, occupancy and mean priority of each tier, h_temporal the recent query rate
# and the session duration.
CACHE_STATE_DIM = 9
TEMPORAL_STATE_DIM = 3


@dataclass
class HMAGConfig:
    """Configuration of an HMAG instance (defaults as in Table 1)."""
    # Answer cache
    l1_cache_size: int = 1500
    l2_cache_size: int = 2100
    l3_cache_size: int = 7500
    similarity_threshold: float = 0.85
    semantic_answer_lookup: bool = True  # False: Retrieval-only reuses verbatim repeats only
    cache_policy: str = "learned"  # "learned" | "lru" | "lfu" | "fixed_temp"
    cache_n_min: int = 1
    priority_weights: Tuple[float, float, float, float] = (0.3, 0.2, 0.2, 0.3)
    promote_threshold: float = 0.5
    demote_threshold: float = 0.3
    # Tier placement: "hot" stores new answers in the lowest tier with free space, promotes on
    # every hit (verbatim or semantic) after the answer is returned, and exchanges a promoted
    # answer with the lowest-priority entry of a full upper tier; "insert_l1" stores new answers
    # in L1 and promotes only into free space (HMAG (original placement)).
    cache_placement: str = "hot"
    cache_frequency_norm: str = "log"  # frequency term of Eq. (3): "log" or "exp"
    cache_promote_to_top: bool = True  # a promoted answer moves straight to L1 (hot placement)
    cache_promote_sweep: bool = False  # periodic promotion/demotion sweep (used with "insert_l1")
    diversity_lambda: float = 0.1
    cache_admission: bool = True  # admission filter of the answer cache (Algorithm 1)
    # Separate single-tier pool of retrieved passages, used only as the cached side of
    # the consistency check, so that passages never displace stored answers.
    document_cache_size: int = 500

    # Retrieval
    retrieval_model: str = "sentence-transformers/all-mpnet-base-v2"
    retrieval_top_k: int = 10
    context_window_tokens: int = 4096
    context_reserve_tokens: int = 512  # reserved for the answer
    use_gpu: bool = torch.cuda.is_available()

    # Controller
    controller_type: str = "thompson_sampling"  # "thompson_sampling" | "policy_gradient" | "fixed"
    fixed_strategy_idx: int = 2  # strategy index for controller_type == "fixed"
    controller_hidden_dim: int = 256
    controller_learning_rate: float = 1e-3
    exploration_rate: float = 0.1
    thompson_sigma: float = 1.0
    thompson_lambda: float = 1.0

    # Reward, Eq. (17): r_t = clip(w_q Q_t - w_l min(2, L_t / L_ref) + w_h chi_t, -1, 1),
    # with L_ref the median latency of the last `reward_latency_window` generations.
    reward_w_quality: float = 0.6
    reward_w_latency: float = 0.3
    reward_w_hitrate: float = 0.1
    reward_latency_window: int = 100

    # Verifier
    consistency_threshold: float = 0.85
    contradiction_threshold: float = 0.7
    verifier_mode: str = "full"  # "full" | "nli_only"

    # Generator
    ollama_model: str = "llama3.2:latest"
    ollama_host: str = "http://localhost:11434"
    generation_temperature: float = 0.7
    max_generation_tokens: int = 1024

    # Ablation switches
    enable_verification: bool = True
    enable_generation: bool = True

    # Monitoring and checkpoints (checkpoints are off unless requested)
    enable_monitoring: bool = True
    monitoring_window: int = 1000
    checkpoint_interval: int = 0  # save every n queries; 0 disables periodic saving
    restore_checkpoint: bool = False
    checkpoint_path: str = "./checkpoints"


@dataclass
class HMAGResponse:
    """Answer to one query with its provenance and timing."""
    response: str
    query: str
    strategy_used: RetrievalStrategy
    documents_used: List[Document]
    consistency_result: Optional[ConsistencyResult]
    latency_breakdown: Dict[str, float]
    total_latency_ms: float
    cache_hit: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Sliding-window statistics of stage latencies, strategy choices and rewards."""

    STAGES = ('cache', 'controller', 'retrieval', 'verification', 'generation',
              'time_to_first_token', 'decoding', 'total')

    def __init__(self, window_size: int = 1000):
        self.latency_windows = {stage: deque(maxlen=window_size) for stage in self.STAGES}
        self.strategy_counts = {strategy: 0 for strategy in RetrievalStrategy}
        self.reward_window = deque(maxlen=window_size)
        self.errors = []
        self.lock = threading.RLock()

    def record_latency(self, component: str, latency_ms: float):
        with self.lock:
            if component in self.latency_windows:
                self.latency_windows[component].append(latency_ms)

    def record_strategy(self, strategy: RetrievalStrategy):
        with self.lock:
            self.strategy_counts[strategy] += 1

    def record_reward(self, reward: float):
        with self.lock:
            self.reward_window.append(reward)

    def get_statistics(self) -> Dict[str, Any]:
        with self.lock:
            total = max(sum(self.strategy_counts.values()), 1)
            return {
                'latency': {
                    stage: {
                        'mean': float(np.mean(w)) if w else 0.0,
                        'p50': float(np.percentile(w, 50)) if w else 0.0,
                        'p95': float(np.percentile(w, 95)) if w else 0.0,
                    }
                    for stage, w in self.latency_windows.items()
                },
                'reward_mean': float(np.mean(self.reward_window)) if self.reward_window else 0.0,
                'strategy_distribution': {str(s): c / total for s, c in self.strategy_counts.items()},
                'errors': len(self.errors),
            }


class HMAGSystem:
    """HMAG pipeline (Section 4)."""

    def __init__(self, config: Optional[HMAGConfig] = None,
                 retriever: Optional[HierarchicalRetriever] = None,
                 generator: Optional[OllamaGenerator] = None,
                 verifier_encoder: Optional[SentenceTransformer] = None,
                 verifier_nli_model=None,
                 verifier_fact_extractor=None):
        """A retriever, generator and the verifier models may be passed in so that several
        instances share one copy of each model; otherwise they are built from `config`."""
        self.config = config or HMAGConfig()

        self.retriever = retriever or HierarchicalRetriever(
            model_name=self.config.retrieval_model, use_gpu=self.config.use_gpu)

        # Both caches use the retriever's encoder and its query-embedding memo, so a query
        # is encoded at most once per request.
        cache_args = dict(
            embedding_dim=self.retriever.embedding_dim,
            similarity_thresholds=(self.config.similarity_threshold,) * 3,
            n_min=self.config.cache_n_min,
            priority_weights=self.config.priority_weights,
            promote_threshold=self.config.promote_threshold,
            demote_threshold=self.config.demote_threshold,
            diversity_lambda=self.config.diversity_lambda,
            encoder=self.retriever.encoder,
            embed_cache=self.retriever._query_embed_cache,
            cache_policy=self.config.cache_policy,
        )
        self.cache = HierarchicalCache(l1_size=self.config.l1_cache_size,
                                       l2_size=self.config.l2_cache_size,
                                       l3_size=self.config.l3_cache_size,
                                       admission_filter=self.config.cache_admission,
                                       placement=self.config.cache_placement,
                                       frequency_norm=self.config.cache_frequency_norm,
                                       promote_to_top=self.config.cache_promote_to_top,
                                       promote_sweep_interval=(25 if self.config.cache_promote_sweep else 10 ** 9),
                                       **cache_args)
        self.doc_cache = HierarchicalCache(l1_size=self.config.document_cache_size,
                                           l2_size=0, l3_size=0, **cache_args)

        self.context_dim = self.retriever.embedding_dim + CACHE_STATE_DIM + TEMPORAL_STATE_DIM
        self.num_actions = len(RetrievalStrategy)
        self.controller = self._build_controller()

        self.verifier = AdvancedVerifier(
            consistency_threshold=self.config.consistency_threshold,
            contradiction_threshold=self.config.contradiction_threshold,
            encoder=verifier_encoder,
            nli_model=verifier_nli_model,
            mode=self.config.verifier_mode,
            fact_extractor=verifier_fact_extractor,
        )

        self.generator = generator or OllamaGenerator(
            model_name=self.config.ollama_model,
            host=self.config.ollama_host,
            temperature=self.config.generation_temperature,
            max_tokens=self.config.max_generation_tokens,
        )

        self.monitor = PerformanceMonitor(self.config.monitoring_window) if self.config.enable_monitoring else None

        # process_query may be called from several threads; the controller updates its
        # posterior (or network) in place and is therefore guarded by a lock.
        self.controller_lock = threading.Lock()
        self._generation_latencies: deque = deque(maxlen=self.config.reward_latency_window)

        self.total_queries = 0
        self.session_start = time.perf_counter()
        self.recent_query_times: deque = deque(maxlen=200)

        if self.config.restore_checkpoint:
            self._load_checkpoint()
        logger.info(f"HMAG initialized (controller={self.config.controller_type})")

    def _build_controller(self):
        if self.config.controller_type == "thompson_sampling":
            return LinearThompsonSamplingController(
                context_dim=self.context_dim, num_actions=self.num_actions,
                sigma=self.config.thompson_sigma, lam=self.config.thompson_lambda)
        if self.config.controller_type == "policy_gradient":
            return PolicyGradientController(
                input_dim=self.context_dim, hidden_dim=self.config.controller_hidden_dim,
                num_actions=self.num_actions, alpha0=self.config.controller_learning_rate,
                epsilon0=self.config.exploration_rate)
        if self.config.controller_type == "fixed":
            return FixedStrategyController(self.config.fixed_strategy_idx)
        raise ValueError(f"Unknown controller_type: {self.config.controller_type}")

    def _maybe_checkpoint(self):
        interval = self.config.checkpoint_interval
        if interval and self.total_queries % interval == 0:
            self._save_checkpoint()

    def process_query(self, query: str, context: Optional[QueryContext] = None) -> HMAGResponse:
        """Answers one query (Algorithm 1 and Section 4.1), then applies the tier promotions of
        its hit (cache maintenance after the answer is returned; its time is reported in
        latency_breakdown['cache_maintenance'] and is not part of total_latency_ms)."""
        result = self._answer(query, context)
        t0 = time.perf_counter()
        self.cache.maintain()
        result.latency_breakdown['cache_maintenance'] = (time.perf_counter() - t0) * 1000
        return result

    def _answer(self, query: str, context: Optional[QueryContext] = None) -> HMAGResponse:
        start_time = time.perf_counter()
        self.total_queries += 1
        self.recent_query_times.append(start_time)

        # A verbatim repeat is answered by a hash lookup of the stored answer, before any
        # encoding or controller decision.
        exact_key = hashlib.sha256(query.encode()).hexdigest()
        exact_entry = self.cache.get(exact_key)
        if exact_entry is not None and exact_entry.entry_type == "qa_response":
            total_latency = (time.perf_counter() - start_time) * 1000
            self.cache.stats.record_search(hit=True, tier_hit=self.cache.tier_of(exact_key))
            if self.monitor:
                self.monitor.record_latency('total', total_latency)
            self._maybe_checkpoint()
            return HMAGResponse(
                response=exact_entry.content,
                query=query,
                strategy_used=RetrievalStrategy.CACHE_ONLY,
                documents_used=[Document(doc_id=exact_key, content=exact_entry.content,
                                         relevance_score=exact_entry.credibility_score,
                                         source_type="cache", metadata={'entry_type': 'qa_response'})],
                consistency_result=None,
                latency_breakdown={'exact_match': total_latency},
                total_latency_ms=total_latency,
                cache_hit=True,
                metadata={
                    'total_queries': self.total_queries,
                    'cache_hit_rate': self.cache.stats.get_recent_hit_rate(),
                    'session_duration': time.perf_counter() - self.session_start,
                    'prompt_tokens': 0, 'generated_tokens': 0,
                    'generation_error': False, 'exact_match': True,
                },
            )

        latency_breakdown = {}
        try:
            # State construction is dominated by the query encoding and is reported as the
            # cache stage, as for the baselines, which encode the query for their lookup.
            state_start = time.perf_counter()
            state = self._extract_state(query)
            latency_breakdown['cache'] = (time.perf_counter() - state_start) * 1000

            controller_start = time.perf_counter()
            with self.controller_lock:
                strategy_idx = self.controller.select_action(state)
            strategy = list(RetrievalStrategy)[min(strategy_idx, self.num_actions - 1)]
            latency_breakdown['controller'] = (time.perf_counter() - controller_start) * 1000
            logger.debug(f"Selected strategy: {strategy}")

            retrieval_start = time.perf_counter()
            if strategy == RetrievalStrategy.RETRIEVAL_ONLY and self.config.semantic_answer_lookup:
                # A stored answer to a matching question is reused; otherwise the answer
                # is generated from fresh corpus documents only.
                documents, cache_hit = self._execute_strategy(query, RetrievalStrategy.CACHE_ONLY)
                if not cache_hit:
                    documents, cache_hit = self._execute_strategy(query, strategy)
            else:
                documents, cache_hit = self._execute_strategy(query, strategy)
            if strategy == RetrievalStrategy.CACHE_ONLY and not cache_hit:
                # Without a matching answer, Cache-only falls back to corpus retrieval.
                documents, _ = self._execute_strategy(query, RetrievalStrategy.RETRIEVAL_ONLY)
            latency_breakdown['retrieval'] = (time.perf_counter() - retrieval_start) * 1000

            # A hit is a stored answer ("qa_response" entry) that matches the question; it is
            # returned without verification, which compares documents rather than answers.
            qa_response_doc = next(
                (d for d in documents if d.metadata.get('entry_type') == 'qa_response'), None)
            reused_cached_response = cache_hit and qa_response_doc is not None

            consistency_result = None
            if (not reused_cached_response and self.config.enable_verification
                    and strategy in (RetrievalStrategy.HYBRID, RetrievalStrategy.ADAPTIVE_HYBRID)):
                verify_start = time.perf_counter()
                consistency_result = self._verify_consistency(query, documents)
                latency_breakdown['verification'] = (time.perf_counter() - verify_start) * 1000
                if not consistency_result.is_consistent:
                    documents = self._resolve_inconsistencies(query, documents, consistency_result)
                if strategy == RetrievalStrategy.ADAPTIVE_HYBRID:
                    documents = self._adaptive_reweight(query, documents, consistency_result)

            generation_start = time.perf_counter()
            if reused_cached_response:
                response = {'response': qa_response_doc.content}
            elif self.config.enable_generation:
                response = self._generate_response(query, documents)
            else:
                response = {'response': ''}
            latency_breakdown['generation'] = (time.perf_counter() - generation_start) * 1000
            if 'ttft_ms' in response:  # prefill and decoding times reported by the server
                latency_breakdown['time_to_first_token'] = response['ttft_ms']
                latency_breakdown['decoding'] = response.get('decoding_ms', 0.0)

            update_start = time.perf_counter()
            self._update_cache(query, documents, response)
            latency_breakdown['cache_update'] = (time.perf_counter() - update_start) * 1000

            total_latency = (time.perf_counter() - start_time) * 1000
            if self.monitor:
                self.monitor.record_latency('total', total_latency)
                self.monitor.record_strategy(strategy)
                for component, latency in latency_breakdown.items():
                    self.monitor.record_latency(component, latency)

            if response.get('error'):
                quality_proxy = 0.0
            elif consistency_result is not None:
                quality_proxy = consistency_result.confidence_score
            else:
                quality_proxy = 1.0
            generated = (not reused_cached_response) and self.config.enable_generation
            reward = self._calculate_reward(total_latency, quality_proxy, cache_hit, generated)
            with self.controller_lock:
                self.controller.update(state, strategy_idx, reward)
            if self.monitor:
                self.monitor.record_reward(reward)
            self._maybe_checkpoint()

            return HMAGResponse(
                response=response['response'],
                query=query,
                strategy_used=strategy,
                documents_used=documents,
                consistency_result=consistency_result,
                latency_breakdown=latency_breakdown,
                total_latency_ms=total_latency,
                cache_hit=cache_hit,
                metadata={
                    'total_queries': self.total_queries,
                    'cache_hit_rate': self.cache.stats.get_recent_hit_rate(),
                    'reward': reward,
                    'session_duration': time.perf_counter() - self.session_start,
                    'prompt_tokens': response.get('prompt_tokens', 0),
                    'generated_tokens': response.get('generated_tokens', 0),
                    'generation_error': bool(response.get('error')),
                },
            )

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            if self.monitor:
                self.monitor.errors.append({'timestamp': time.time(), 'query': query, 'error': str(e)})
            return HMAGResponse(
                response=f"Error: {e}",
                query=query,
                strategy_used=RetrievalStrategy.FALLBACK,
                documents_used=[],
                consistency_result=None,
                latency_breakdown=latency_breakdown,
                total_latency_ms=(time.perf_counter() - start_time) * 1000,
                cache_hit=False,
                metadata={'generation_error': True},
            )

    def _extract_state(self, query: str) -> np.ndarray:
        """Controller state x_t = [phi(q_t), h_cache, h_temporal] (Algorithm 2)."""
        query_embedding = self.cache.encode_cached(query)
        now = time.perf_counter()
        h_cache = np.array(
            [self.cache.stats.get_tier_hit_rate(i) for i in range(3)]
            + [len(self.cache.tiers[i]) / max(self.cache.capacities[i], 1) for i in range(3)]
            + [self._tier_avg_priority(i, self.cache.clock()) for i in range(3)])

        recent = list(self.recent_query_times)[-20:]
        recent_rate = len(recent) / max(recent[-1] - recent[0], 1e-6) if len(recent) >= 2 else 0.0
        session_hours = (now - self.session_start) / 3600.0
        # The second temporal slot is a constant 0; it keeps the state dimension of the
        # controller fixed.
        h_temporal = np.array([recent_rate, 0.0, session_hours])

        return np.concatenate([query_embedding, h_cache, h_temporal]).astype('float32')

    def _tier_avg_priority(self, tier_idx: int, now: float, sample_size: int = 200) -> float:
        """Mean priority of a tier (feature T-bar_i), estimated from a random sample of at
        most `sample_size` entries to bound the cost per query."""
        tier = self.cache.tiers[tier_idx]
        if not tier:
            return 0.0
        entries = list(tier.values())
        if len(entries) > sample_size:
            idx = np.random.choice(len(entries), size=sample_size, replace=False)
            entries = [entries[i] for i in idx]
        return float(np.mean([self.cache._priority(e, now) for e in entries]))

    def _execute_strategy(self, query: str, strategy: RetrievalStrategy) -> Tuple[List[Document], bool]:
        """Runs one retrieval strategy (Eq. (6)) and returns (documents, cache_hit)."""
        if strategy == RetrievalStrategy.CACHE_ONLY:
            entries, hit, _ = self.cache.search(query, k=self.config.retrieval_top_k,
                                                entry_type_filter="qa_response")
            documents = [Document(doc_id=e.key, content=e.content, relevance_score=e.credibility_score,
                                  source_type="cache", metadata={'entry_type': e.entry_type})
                         for e in entries]
            return documents, hit

        if strategy == RetrievalStrategy.RETRIEVAL_ONLY:
            result = asyncio.run(self.retriever.retrieve_async(
                query, k=self.config.retrieval_top_k, strategy="dense"))
            return result.documents, False

        if strategy in (RetrievalStrategy.HYBRID, RetrievalStrategy.ADAPTIVE_HYBRID):
            cached_docs, cache_hit = self._execute_strategy(query, RetrievalStrategy.CACHE_ONLY)
            if cache_hit:
                # The stored answer is reused, so no documents are needed.
                return cached_docs, cache_hit

            retrieved_docs, _ = self._execute_strategy(query, RetrievalStrategy.RETRIEVAL_ONLY)
            cached_doc_entries, _, _ = self.doc_cache.search(query, k=self.config.retrieval_top_k)
            # Cached passages are scored by their cosine similarity to the query, the scale of
            # the retrieval scores, and the merged list keeps the k highest-scoring documents.
            q_emb = self.cache.encode_cached(query)
            q_emb = q_emb / (np.linalg.norm(q_emb) or 1.0)
            cached_context_docs = []
            for e in cached_doc_entries:
                e_emb = e.embedding / (np.linalg.norm(e.embedding) or 1.0)
                cached_context_docs.append(Document(
                    doc_id=e.key, content=e.content, relevance_score=float(np.dot(q_emb, e_emb)),
                    source_type="cache", metadata={'entry_type': e.entry_type}))

            all_docs = {}
            for doc in cached_docs + cached_context_docs + retrieved_docs:
                if doc.doc_id not in all_docs or doc.relevance_score > all_docs[doc.doc_id].relevance_score:
                    all_docs[doc.doc_id] = doc
            merged = sorted(all_docs.values(), key=lambda d: d.relevance_score, reverse=True)
            return merged[:self.config.retrieval_top_k], cache_hit

        if strategy == RetrievalStrategy.HIERARCHICAL:
            # Tier-by-tier cache search, then corpus retrieval on a miss.
            cached_docs, cache_hit = self._execute_strategy(query, RetrievalStrategy.CACHE_ONLY)
            if cache_hit:
                return cached_docs, True
            retrieved_docs, _ = self._execute_strategy(query, RetrievalStrategy.RETRIEVAL_ONLY)
            return retrieved_docs, False

        if strategy == RetrievalStrategy.FALLBACK:
            # Cache search, then a corpus search over twice the usual number of documents.
            cached_docs, cache_hit = self._execute_strategy(query, RetrievalStrategy.CACHE_ONLY)
            if cache_hit:
                return cached_docs, True
            result = asyncio.run(self.retriever.retrieve_async(
                query, k=self.config.retrieval_top_k * 2, strategy="dense"))
            return self._fit_context(query, result.documents), False

        return self._execute_strategy(query, RetrievalStrategy.RETRIEVAL_ONLY)

    def _fit_context(self, query: str, documents: List[Document]) -> List[Document]:
        """Keeps the top-k documents and adds further ones only while the prompt fits the
        context window. Token counts are estimated from the character length of the text
        the generator includes (at most 1,000 characters per document, about 3 characters
        per token)."""
        k = self.config.retrieval_top_k
        budget = self.config.context_window_tokens - self.config.context_reserve_tokens
        used = (len(query) + 400) / 3.0  # instruction, question and prompt template
        kept = []
        for i, doc in enumerate(documents):
            cost = (min(len(doc.content), 1000) + 40) / 3.0
            if i >= k and used + cost > budget:
                break
            kept.append(doc)
            used += cost
        return kept

    def _adaptive_reweight(self, query: str, documents: List[Document],
                           consistency_result: ConsistencyResult) -> List[Document]:
        """Adaptive-hybrid strategy: scales the scores of cached documents by the
        consistency score and a query-length proxy for complexity."""
        complexity = min(1.0, len(query.split()) / 30.0)
        for doc in documents:
            if doc.source_type == "cache":
                doc.relevance_score *= (0.5 + 0.5 * consistency_result.confidence_score) * (1.0 - 0.3 * complexity)
        documents.sort(key=lambda d: d.relevance_score, reverse=True)
        return documents

    def _verify_consistency(self, query: str, documents: List[Document]) -> ConsistencyResult:
        cached_docs = [doc.content for doc in documents if doc.source_type == "cache"]
        retrieved_docs = [doc.content for doc in documents if doc.source_type != "cache"]
        if not cached_docs or not retrieved_docs:
            return ConsistencyResult(is_consistent=True, confidence_score=1.0, contradictions=[],
                                     supporting_evidence=[], resolution_strategy="no_check_needed",
                                     latency_ms=0)
        return self.verifier.verify_consistency(cached_docs, retrieved_docs, query)

    def _resolve_inconsistencies(self, query: str, documents: List[Document],
                                 consistency_result: ConsistencyResult) -> List[Document]:
        strategy = consistency_result.resolution_strategy
        if strategy == "prefer_recent":
            documents.sort(key=lambda x: x.timestamp, reverse=True)
        elif strategy in ("retrieval_with_verification", "full_retrieval"):
            documents, _ = self._execute_strategy(query, RetrievalStrategy.RETRIEVAL_ONLY)
        elif strategy == "weighted_merge":
            for doc in documents:
                if doc.source_type == "cache":
                    doc.relevance_score *= consistency_result.confidence_score
            documents.sort(key=lambda d: d.relevance_score, reverse=True)
        return documents[:self.config.retrieval_top_k]

    def _generate_response(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        return self.generator.generate(prompt=query, context=documents, system_prompt=SYSTEM_PROMPT)

    def _update_cache(self, query: str, documents: List[Document], response: Dict[str, Any]):
        """Stores the answer under the embedding of the question as asked (also after a
        hit, so a reused answer can be found from either wording) and adds the top
        retrieved passages to the document cache."""
        if response.get('error'):
            return
        self.cache.put(
            key=hashlib.sha256(query.encode()).hexdigest(),
            content=response['response'],
            embedding_text=query,
            entry_type="qa_response",
            metadata={'query': query, 'timestamp': time.time(), 'documents_used': len(documents)},
        )
        for doc in documents[:3]:
            if doc.source_type != "cache":
                self.doc_cache.put(key=doc.doc_id, content=doc.content,
                                   metadata={'relevance_score': doc.relevance_score, 'source_type': 'retrieval'})

    def _calculate_reward(self, latency_ms: float, quality_proxy: float, cache_hit: bool,
                          generated: bool = False) -> float:
        """Reward of Eq. (17). Q_t is the verifier's agreement score when verification ran,
        0 for a failed generation and 1 otherwise; L_ref is the median latency of recent
        queries that required generation."""
        if generated:
            self._generation_latencies.append(latency_ms)
        latency_ref = float(np.median(self._generation_latencies)) if self._generation_latencies else max(latency_ms, 1.0)
        latency_term = min(2.0, latency_ms / max(latency_ref, 1.0))
        reward = (self.config.reward_w_quality * quality_proxy
                  - self.config.reward_w_latency * latency_term
                  + self.config.reward_w_hitrate * (1.0 if cache_hit else 0.0))
        return float(np.clip(reward, -1.0, 1.0))

    def _save_checkpoint(self):
        path = Path(self.config.checkpoint_path)
        path.mkdir(exist_ok=True)
        controller_state = self.controller.state_dict() if isinstance(self.controller, PolicyGradientController) else None
        checkpoint = {
            'controller_type': self.config.controller_type,
            'controller_state': controller_state,
            'cache_stats': self.cache.stats.cache_stats,
            'total_queries': self.total_queries,
            'session_start': self.session_start,
        }
        torch.save(checkpoint, path / f"checkpoint_{self.total_queries}.pt")

    def _load_checkpoint(self):
        checkpoints = list(Path(self.config.checkpoint_path).glob("checkpoint_*.pt"))
        if not checkpoints:
            return
        latest = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
        try:
            checkpoint = torch.load(latest, weights_only=False)
        except Exception as e:
            logger.warning(f"Could not load checkpoint {latest}: {e}")
            return
        if checkpoint.get('controller_type') != self.config.controller_type:
            logger.info("Checkpoint controller type differs from the configuration; not restored.")
            return
        if checkpoint.get('controller_state') is not None and isinstance(self.controller, PolicyGradientController):
            self.controller.load_state_dict(checkpoint['controller_state'])
        self.cache.stats.cache_stats = checkpoint['cache_stats']
        self.total_queries = checkpoint['total_queries']
        self.session_start = checkpoint['session_start']
        logger.info(f"Loaded checkpoint {latest}")

    def shutdown(self):
        if self.config.checkpoint_interval:
            self._save_checkpoint()
