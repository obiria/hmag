"""Hierarchical answer cache (Algorithm 1) and the retrieval-strategy controllers:
linear Thompson sampling (Algorithm 2, Theorem 3), policy gradient (Algorithm 3,
Theorem 2) and a fixed-strategy controller for ablations."""

import logging
import math
import os
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """The six retrieval strategies of Eq. (6)."""
    CACHE_ONLY = "cache_only"
    RETRIEVAL_ONLY = "retrieval_only"
    HYBRID = "hybrid"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE_HYBRID = "adaptive_hybrid"
    FALLBACK = "fallback"


@dataclass
class QueryContext:
    query: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """Cache entry. `entry_type` is "qa_response" for a stored answer (embedded by its
    question) or "document" for a passage (embedded by its content)."""
    key: str
    content: str
    embedding: np.ndarray  # L2-normalized
    timestamp: float
    last_accessed: float = 0.0
    credibility_score: float = 1.0
    access_times: deque = field(default_factory=lambda: deque(maxlen=64))
    frozen_priority: Optional[float] = None  # used by cache_policy="fixed_temp"
    entry_type: str = "document"

    def record_access(self, t: float):
        self.access_times.append(t)
        self.last_accessed = t

    @property
    def access_count(self) -> int:
        return len(self.access_times)


class MemoryStatistics:
    """Hit rates over a sliding window of lookups, overall and per tier."""

    def __init__(self, num_tiers: int = 3, window_size: int = 1000):
        self.num_tiers = num_tiers
        self.hit_rate_window = deque(maxlen=window_size)
        self.tier_hit_windows = [deque(maxlen=window_size) for _ in range(num_tiers)]
        self._lock = threading.Lock()
        self.cache_stats = {'total_hits': 0, 'total_misses': 0, 'total_evictions': 0}

    def record_search(self, hit: bool, tier_hit: Optional[int]):
        """`tier_hit` is the 0-indexed tier that served the hit, or None on a miss."""
        with self._lock:
            self.hit_rate_window.append(1 if hit else 0)
            for i in range(self.num_tiers):
                self.tier_hit_windows[i].append(1 if (hit and tier_hit == i) else 0)
            self.cache_stats['total_hits' if hit else 'total_misses'] += 1

    def record_eviction(self):
        with self._lock:
            self.cache_stats['total_evictions'] += 1

    def get_recent_hit_rate(self) -> float:
        w = self.hit_rate_window
        return sum(w) / len(w) if w else 0.0

    def get_tier_hit_rate(self, tier_idx: int) -> float:
        w = self.tier_hit_windows[tier_idx]
        return sum(w) / len(w) if w else 0.0


class SlotIndex:
    """Exact inner-product index over fixed slots: an entry is written into a free slot and
    removed by zeroing it, both in O(d), so the index is never rebuilt. An empty slot holds
    the zero vector; the caller filters by its similarity threshold (> 0), so empty slots are
    never returned. The slot array grows in chunks up to the largest occupancy reached."""

    CHUNK = 256

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.key_of: List[Optional[str]] = []
        self.slot_of: Dict[str, int] = {}
        self.free: List[int] = []
        self.rows = None

    def _grow(self):
        n = self.index.ntotal
        self.index.add(np.zeros((self.CHUNK, self.dim), dtype="float32"))
        self.rows = faiss.rev_swig_ptr(self.index.get_xb(), self.index.ntotal * self.dim).reshape(-1, self.dim)
        self.key_of.extend([None] * self.CHUNK)
        self.free.extend(range(n + self.CHUNK - 1, n - 1, -1))

    def add(self, key: str, embedding: np.ndarray):
        if not self.free:
            self._grow()
        slot = self.free.pop()
        self.rows[slot] = embedding
        self.key_of[slot], self.slot_of[key] = key, slot

    def remove(self, key: str):
        slot = self.slot_of.pop(key, None)
        if slot is None:
            return
        self.rows[slot] = 0.0
        self.key_of[slot] = None
        self.free.append(slot)

    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if not self.slot_of:
            return []
        scores, slots = self.index.search(query.reshape(1, -1).astype("float32"), min(k, self.index.ntotal))
        return [(self.key_of[s], float(d)) for d, s in zip(scores[0], slots[0])
                if s >= 0 and self.key_of[s] is not None]


class RequestFrequency:
    """Decayed request counts per key, kept for every requested key whether or not it is
    cached (TinyLFU-style admission, Einziger et al., 2017). After `sample_size` recorded
    requests every count is halved and zero counts are dropped, so the counts follow
    recent popularity and at most O(sample_size) keys are stored. Recording is O(1)
    amortized."""

    def __init__(self, sample_size: int):
        self.sample_size = max(int(sample_size), 1)
        self.counts: Dict[str, int] = {}
        self._recorded = 0

    def record(self, key: str):
        self.counts[key] = self.counts.get(key, 0) + 1
        self._recorded += 1
        if self._recorded >= self.sample_size:
            self._recorded = 0
            self.counts = {k: c // 2 for k, c in self.counts.items() if c > 1}

    def estimate(self, key: str) -> int:
        return self.counts.get(key, 0)


class HierarchicalCache:
    """Three-tier semantic cache (Algorithm 1): threshold-gated search from L1 to L3,
    priority score of Eq. (3), promotion and demotion with hysteresis, and
    diversity-aware eviction (Eq. (16))."""

    POLICIES = ("learned", "lru", "lfu", "fixed_temp")
    EXCHANGE_SAMPLE = 64  # hot placement: the exchange partner is the lowest-priority entry of a sample

    def __init__(self,
                 l1_size: int = 100,
                 l2_size: int = 1000,
                 l3_size: int = 10000,
                 embedding_dim: int = 768,
                 similarity_thresholds: Tuple[float, float, float] = (0.85, 0.85, 0.85),
                 n_min: int = 1,
                 priority_weights: Tuple[float, float, float, float] = (0.3, 0.2, 0.2, 0.3),
                 promote_threshold: float = 0.7,
                 demote_threshold: float = 0.3,
                 recency_lambda: float = 1e-4,
                 freq_tau: float = 3600.0,
                 freq_rho: float = 1.0,
                 diversity_lambda: float = 0.1,
                 recent_query_window: int = 50,
                 promote_sweep_interval: int = 25,
                 encoder: Optional[SentenceTransformer] = None,
                 embed_cache: Optional["OrderedDict[str, np.ndarray]"] = None,
                 cache_policy: str = "learned",
                 admission_filter: bool = False,
                 admission_sample_factor: int = 10,
                 clock: Callable[[], float] = time.monotonic,
                 placement: str = "insert_l1",
                 frequency_norm: str = "exp",
                 search_index: str = "per_tier",
                 directory_train: Optional[np.ndarray] = None,
                 directory_nprobe: int = 16,
                 promote_to_top: bool = False):
        """`cache_policy` selects the priority used for promotion, demotion and eviction:
        "learned" is Eq. (3), recomputed at every use; "lru" is the recency term alone;
        "lfu" the frequency term alone; "fixed_temp" is Eq. (3) computed once at insertion.
        `embed_cache` is a memo of raw query embeddings that can be shared with the
        retriever, so that a query is encoded once per request.
        With `admission_filter`, a new entry enters a full cache only if its decayed
        request count exceeds that of the entry it would displace (Algorithm 1, admission);
        counts are halved every `admission_sample_factor` x total capacity requests.
        All tiers are kept in main memory.
        `placement` selects where answers live: "insert_l1" stores a new answer in L1 and
        promotes only into a tier with free space; "hot" stores a new answer in the lowest
        tier with free space (L3 once the cache is full), counts every hit (verbatim or semantic) towards promotion, and promotes into a
        full tier by exchanging places with its lowest-priority entry when the promoted entry
        has the higher priority, so that L1 holds the highest-priority answers.
        `frequency_norm` maps the frequency score f of Eq. (3) to [0, 1]: "exp" is 1 - exp(-f),
        which is 0.95 after three requests; "log" is log(1 + f) / log(1 + f_max), where f_max
        is the largest frequency score observed so far, so that ranks are kept for popular
        answers.
        `search_index` selects how stored answers are searched: "per_tier" keeps an exact
        index per tier, searched from L1 to L3; "directory" keeps one inverted-file index over
        all answers (trained on `directory_train`, probing `directory_nprobe` lists, exact
        similarities within the probed lists), whose candidates name their tier, so that a
        lookup probes once and a miss needs no scan of the tiers. Among the candidates above
        the threshold, the matches of the uppermost tier are returned, as in the tier order of
        Algorithm 1. Moving an entry between tiers does not change the directory.
        `clock` returns the current time in seconds; the default is monotonic, so that clock
        adjustments do not distort the recency and frequency terms (replaceable for trace replay)."""
        if cache_policy not in self.POLICIES:
            raise ValueError(f"Unknown cache_policy: {cache_policy}")
        if placement not in ("insert_l1", "hot"):
            raise ValueError(f"Unknown placement: {placement}")
        self.placement = placement
        self.promote_to_top = promote_to_top  # hot placement: a promoted answer goes straight to L1
        self._pending_promotions: List[Tuple[CacheEntry, int]] = []  # hot placement: applied by maintain()
        if frequency_norm not in ("exp", "log"):
            raise ValueError(f"Unknown frequency_norm: {frequency_norm}")
        self.frequency_norm = frequency_norm
        self._f_max = 1.0
        if search_index not in ("per_tier", "directory"):
            raise ValueError(f"Unknown search_index: {search_index}")
        self.search_index = search_index
        self.directory = None
        if search_index == "directory":
            if directory_train is None:
                raise ValueError("search_index='directory' needs directory_train")
            total = max(sum((l1_size, l2_size, l3_size)), 1)
            nlist = max(16, min(int(4 * math.sqrt(total)), len(directory_train) // 39))
            self._dir_quantizer = faiss.IndexFlatIP(embedding_dim)
            self.directory = faiss.IndexIVFFlat(self._dir_quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.directory.train(np.ascontiguousarray(directory_train, dtype="float32"))
            self.directory.set_direct_map_type(faiss.DirectMap.Hashtable)
            self.directory.nprobe = directory_nprobe
            self._dir_id, self._dir_key, self._dir_next = {}, {}, 0
        self.tiers: List[OrderedDict] = [OrderedDict(), OrderedDict(), OrderedDict()]
        self.capacities = [l1_size, l2_size, l3_size]
        self.thresholds = list(similarity_thresholds)
        self.n_min = n_min
        self.cache_policy = cache_policy

        self.alpha, self.beta, self.gamma, self.delta = priority_weights
        self.promote_threshold = promote_threshold
        self.demote_threshold = demote_threshold
        self.recency_lambda = recency_lambda
        self.freq_tau = freq_tau
        self.freq_rho = freq_rho
        self.diversity_lambda = diversity_lambda

        self.embedding_dim = embedding_dim
        self.encoder = encoder or SentenceTransformer('all-mpnet-base-v2')
        self._embed_cache: "OrderedDict[str, np.ndarray]" = embed_cache if embed_cache is not None else OrderedDict()
        self._embed_cache_max = 2000

        # Exact inner-product index per tier, updated in place when an entry enters or leaves.
        self.indexes = [SlotIndex(embedding_dim) for _ in range(3)]
        self._dirty = [False, False, False]  # kept for callers of the former rebuild interface

        self.recent_queries: deque = deque(maxlen=recent_query_window)
        self.stats = MemoryStatistics(num_tiers=3)
        self.lock = threading.RLock()
        self.promote_sweep_interval = promote_sweep_interval
        self._puts_since_sweep = 0

        self.clock = clock
        self.admission_filter = admission_filter
        self.frequency = RequestFrequency(admission_sample_factor * max(sum(self.capacities), 1))
        self.rejected_admissions = 0

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        return (embedding / norm).astype('float32') if norm > 0 else embedding.astype('float32')

    def encode_cached(self, text: str) -> np.ndarray:
        """Normalized embedding of `text`; raw embeddings are memoized (LRU, 2,000 texts)."""
        cached = self._embed_cache.get(text)
        if cached is not None:
            self._embed_cache.move_to_end(text)
            return self._normalize(cached)
        raw = self.encoder.encode(text, convert_to_numpy=True)
        self._embed_cache[text] = raw
        if len(self._embed_cache) > self._embed_cache_max:
            self._embed_cache.popitem(last=False)
        return self._normalize(raw)

    @staticmethod
    def _vec(entry: CacheEntry) -> np.ndarray:
        return entry.embedding

    def ram_bytes(self) -> int:
        """Memory held by stored embeddings and contents (entry statistics excluded):
        float32 embedding plus UTF-8 content per entry."""
        return sum(e.embedding.nbytes + len(e.content.encode('utf-8')) for tier in self.tiers for e in tier.values())

    # Priority score, Eq. (3)
    def _recency_score(self, entry: CacheEntry, t: float) -> float:
        return math.exp(-self.recency_lambda * max(0.0, t - entry.last_accessed))

    def _frequency_score(self, entry: CacheEntry, t: float) -> float:
        # Iterates over a copy, since other request threads may append concurrently.
        return sum((1.0 + max(0.0, t - t_i) / self.freq_tau) ** (-self.freq_rho)
                   for t_i in list(entry.access_times))

    def _semantic_importance(self, entry: CacheEntry) -> float:
        recent = list(self.recent_queries)
        if not recent:
            return 0.0
        embedding = self._vec(entry)
        return max(0.0, float(np.mean([float(np.dot(embedding, q)) for q in recent])))

    def _priority(self, entry: CacheEntry, t: float) -> float:
        if self.cache_policy == "fixed_temp":
            if entry.frozen_priority is None:
                entry.frozen_priority = self._learned_priority(entry, t)
            return entry.frozen_priority
        if self.cache_policy == "lru":
            return self._recency_score(entry, t)
        if self.cache_policy == "lfu":
            return 1.0 - math.exp(-self._frequency_score(entry, t))
        return self._learned_priority(entry, t)

    def _frequency_norm(self, entry: CacheEntry, t: float) -> float:
        f = self._frequency_score(entry, t)
        if self.frequency_norm == "log":
            self._f_max = max(self._f_max, f)
            return math.log1p(f) / math.log1p(self._f_max)
        return 1.0 - math.exp(-f)

    def _learned_priority(self, entry: CacheEntry, t: float) -> float:
        f_norm = self._frequency_norm(entry, t)  # frequency mapped to [0, 1]
        return (self.alpha * f_norm + self.beta * self._recency_score(entry, t)
                + self.gamma * self._semantic_importance(entry) + self.delta * entry.credibility_score)

    # Tier bookkeeping
    def _tier_add(self, tier_idx: int, key: str, entry: CacheEntry):
        self.tiers[tier_idx][key] = entry
        if self.directory is None:
            self.indexes[tier_idx].add(key, self._vec(entry))
        if self.directory is not None and key not in self._dir_id:
            i = self._dir_next
            self._dir_next += 1
            self.directory.add_with_ids(self._vec(entry).reshape(1, -1).astype("float32"), np.array([i], dtype="int64"))
            self._dir_id[key], self._dir_key[i] = i, key

    def _directory_remove(self, key: str):
        if self.directory is not None and key in self._dir_id:
            i = self._dir_id.pop(key)
            del self._dir_key[i]
            self.directory.remove_ids(faiss.IDSelectorArray(np.array([i], dtype="int64")))

    def _tier_remove(self, tier_idx: int, key: str) -> CacheEntry:
        entry = self.tiers[tier_idx].pop(key)
        if self.directory is None:
            self.indexes[tier_idx].remove(key)
        return entry

    def _rebuild_index(self, tier_idx: int):
        """No-op: the slot indexes are updated in place."""

    def search(self, query: str, k: int = 5,
               entry_type_filter: Optional[str] = None) -> Tuple[List[CacheEntry], bool, Optional[int]]:
        """Hierarchical search (Algorithm 1, lines 1-7): searches L1, L2 and L3 in turn and
        stops at the first tier with at least n_min entries above its similarity threshold.
        With `entry_type_filter="qa_response"` only stored answers count as matches.
        Returns (matches, hit, tier index)."""
        with self.lock:
            query_embedding = self.encode_cached(query)
            self.recent_queries.append(query_embedding)
            now = self.clock()
            if self.directory is not None:
                return self._directory_search(query_embedding, k, entry_type_filter, now)
            for tier_idx in range(3):
                tier = self.tiers[tier_idx]
                if not tier:
                    continue
                matches = self._tier_matches(tier_idx, query_embedding, k, entry_type_filter)
                if len(matches) >= self.n_min:
                    for entry, _ in matches:
                        entry.record_access(now)
                    if self.admission_filter:
                        self.frequency.record(matches[0][0].key)  # demand for the serving answer
                    self.stats.record_search(hit=True, tier_hit=tier_idx)
                    if tier_idx > 0:
                        if self.placement == "hot":
                            self._pending_promotions.append((matches[0][0], tier_idx))
                        else:
                            self._maybe_promote_hot_entry(matches[0][0], tier_idx)
                    return [view for _, view in matches], True, tier_idx
            self.stats.record_search(hit=False, tier_hit=None)
            return [], False, None

    def _directory_search(self, query_embedding: np.ndarray, k: int, entry_type_filter: Optional[str],
                          now: float) -> Tuple[List[CacheEntry], bool, Optional[int]]:
        """Search through the directory: one probe; the matches of the uppermost tier that
        holds a candidate above its threshold are returned (hit handling as in search)."""
        found: List[List[Tuple[float, CacheEntry]]] = [[], [], []]
        if self._dir_id:
            scores, ids = self.directory.search(query_embedding.reshape(1, -1).astype("float32"), k)
            for score, i in zip(scores[0], ids[0]):
                key = self._dir_key.get(int(i))
                if key is None:
                    continue
                t = self.tier_of(key)
                if t is None or score < self.thresholds[t]:
                    continue
                entry = self.tiers[t][key]
                if entry_type_filter is None or entry.entry_type == entry_type_filter:
                    found[t].append((float(score), entry))
        for tier_idx in range(3):
            if len(found[tier_idx]) >= self.n_min:
                matches = [e for _, e in sorted(found[tier_idx], key=lambda x: -x[0])]
                for entry in matches:
                    entry.record_access(now)
                if self.admission_filter:
                    self.frequency.record(matches[0].key)
                self.stats.record_search(hit=True, tier_hit=tier_idx)
                if tier_idx > 0:
                    if self.placement == "hot":
                        self._pending_promotions.append((matches[0], tier_idx))
                    else:
                        self._maybe_promote_hot_entry(matches[0], tier_idx)
                return matches, True, tier_idx
        self.stats.record_search(hit=False, tier_hit=None)
        return [], False, None

    def _tier_matches(self, tier_idx: int, query_embedding: np.ndarray, k: int,
                      entry_type_filter: Optional[str]) -> List[Tuple[CacheEntry, CacheEntry]]:
        """Entries of a tier with similarity at least the tier threshold, best first, as
        (entry, view) pairs, where the view carries the content."""
        tier = self.tiers[tier_idx]
        threshold = self.thresholds[tier_idx]
        out = []
        for key, score in self.indexes[tier_idx].search(query_embedding, min(k, len(tier))):
            entry = tier.get(key)
            if entry is not None and score >= threshold and (
                    entry_type_filter is None or entry.entry_type == entry_type_filter):
                out.append((entry, entry))
        return out

    def _maybe_promote_hot_entry(self, entry: CacheEntry, tier_idx: int):
        """Promotes a hit entry one tier up when its priority reaches the promotion
        threshold and the upper tier has room."""
        now = self.clock()
        priority = self._priority(entry, now)
        if priority < self.promote_threshold:
            return
        target_idx = 0 if (self.promote_to_top and self.placement == "hot") else tier_idx - 1
        while target_idx > 0 and self.capacities[target_idx] == 0:
            target_idx -= 1
        if self.capacities[target_idx] == 0:
            return
        if len(self.tiers[target_idx]) >= self.capacities[target_idx]:
            if self.placement != "hot":
                return
            items = list(self.tiers[target_idx].items())
            if len(items) > self.EXCHANGE_SAMPLE:
                items = [items[i] for i in np.random.choice(len(items), self.EXCHANGE_SAMPLE, replace=False)]
            victim_key, victim = min(items, key=lambda kv: self._priority(kv[1], now))
            if self._priority(victim, now) >= priority:
                return
            self._tier_add(tier_idx, victim_key, self._tier_remove(target_idx, victim_key))
        self._tier_remove(tier_idx, entry.key)
        self._tier_add(target_idx, entry.key, entry)

    def put(self, key: str, content: str, embedding_text: Optional[str] = None,
            entry_type: str = "document", metadata: Optional[Dict] = None):
        """Inserts an entry into L1 and restores the tier capacities. The entry is embedded
        from `embedding_text` (the question, for a stored answer) or from `content`.
        A full promotion and demotion sweep runs every `promote_sweep_interval` insertions;
        entries that are hit are promoted immediately during search. With the admission
        filter, a new key is not stored when the cache is full and its request count does
        not exceed that of the entry the eviction rule would remove."""
        with self.lock:
            if self._pending_promotions:
                self.maintain()
            now = self.clock()
            if self.admission_filter and not self._admits(key, now):
                self.rejected_admissions += 1
                return
            entry = CacheEntry(
                key=key,
                content=content,
                embedding=self.encode_cached(embedding_text or content),
                timestamp=now,
                credibility_score=(metadata or {}).get('credibility', 1.0),
                entry_type=entry_type,
            )
            entry.record_access(now)
            home = 0
            if self.placement == "hot":  # lowest tier with free space, L3 once the cache is full
                free = [i for i in range(3) if 0 < self.capacities[i] and len(self.tiers[i]) < self.capacities[i]]
                home = max(free) if free else self._exit_tier()
            for tier_idx, tier in enumerate(self.tiers):
                if key in tier:
                    if self.placement == "hot":
                        home = tier_idx  # an answer already cached keeps its tier
                    self._tier_remove(tier_idx, key)
            self._tier_add(home, key, entry)
            self.demote()

            self._puts_since_sweep += 1
            if self._puts_since_sweep >= self.promote_sweep_interval:
                self._puts_since_sweep = 0
                self.promote()
                self.demote_stale()

    def _exit_tier(self) -> int:
        """Lowest tier with nonzero capacity; its eviction victims leave the cache."""
        return max(i for i, c in enumerate(self.capacities) if c > 0)

    def _admits(self, key: str, now: float) -> bool:
        """Admission test. Keys already cached and inserts into a cache with free space are
        always admitted; otherwise the candidate must have a strictly higher request count
        than the entry that would leave the cache."""
        if any(key in tier for tier in self.tiers):
            return True
        if sum(len(tier) for tier in self.tiers) < sum(self.capacities):
            return True
        last = self._exit_tier()
        if not self.tiers[last]:
            return True
        _, victim = self._pick_eviction_victim(last, now)
        return self.frequency.estimate(key) > self.frequency.estimate(victim.key)

    def promote(self):
        """Promotion sweep (Algorithm 1): moves entries with priority at least the promotion
        threshold from L3 to L2 and from L2 to L1 while the upper tier has room."""
        with self.lock:
            now = self.clock()
            for tier_idx in (2, 1):
                target_idx = tier_idx - 1
                tier = self.tiers[tier_idx]
                candidates = [key for key, e in tier.items() if self._priority(e, now) >= self.promote_threshold]
                for key in candidates:
                    if len(self.tiers[target_idx]) >= self.capacities[target_idx]:
                        break
                    self._tier_add(target_idx, key, self._tier_remove(tier_idx, key))

    def demote_stale(self):
        """Demotion sweep: moves entries with priority at most the demotion threshold one
        tier down. Entries are not demoted when no lower tier has capacity."""
        with self.lock:
            now = self.clock()
            for tier_idx in (0, 1):
                target_idx = tier_idx + 1
                if sum(self.capacities[target_idx:]) == 0:
                    continue
                tier = self.tiers[tier_idx]
                candidates = [key for key, e in tier.items() if self._priority(e, now) <= self.demote_threshold]
                for key in candidates:
                    self._push_down(target_idx, key, self._tier_remove(tier_idx, key), now)

    def _pick_eviction_victim(self, tier_idx: int, now: float, sample_size: int = 200) -> Tuple[str, CacheEntry]:
        """Eviction victim of Eq. (16): minimizes priority + lambda * (largest similarity to
        another candidate) among the 2 * overflow (at least 10) lowest-priority entries of a
        random sample of at most `sample_size` entries. With the admission filter, the victim
        of the last tier, which leaves the cache, is the sampled entry with the lowest request
        count. The tier must be non-empty."""
        tier = self.tiers[tier_idx]
        overflow = max(len(tier) - self.capacities[tier_idx], 1)
        items = list(tier.items())
        if len(items) > sample_size:
            idx = np.random.choice(len(items), size=sample_size, replace=False)
            items = [items[i] for i in idx]
        if self.admission_filter and tier_idx == self._exit_tier():
            # An entry that leaves the cache is the least requested one, ties broken by
            # priority; request counts persist after eviction, unlike access histories.
            return min(items, key=lambda kv: (self.frequency.estimate(kv[0]), self._priority(kv[1], now)))
        candidate_pool = sorted(items, key=lambda kv: self._priority(kv[1], now))[:max(overflow * 2, 10)]
        pool_embeddings = np.stack([self._vec(e) for _, e in candidate_pool])

        best_key, best_score = None, float('inf')
        for key, entry in candidate_pool:
            sims = pool_embeddings @ self._vec(entry)
            others = sims[sims < 0.999]  # excludes the entry itself and exact duplicates
            score = self._priority(entry, now) + self.diversity_lambda * (float(np.max(others)) if others.size else 0.0)
            if score < best_score:
                best_score, best_key = score, key
        return best_key, tier[best_key]

    def _push_down(self, tier_idx: int, key: str, entry: CacheEntry, now: float):
        """Inserts an entry into `tier_idx`; if the tier is full, its eviction victim moves
        down first. Past L3 the entry leaves the cache."""
        if tier_idx >= len(self.tiers):
            self.stats.record_eviction()
            self._directory_remove(key)
            return
        tier = self.tiers[tier_idx]
        if len(tier) >= self.capacities[tier_idx]:
            if not tier:  # tier of capacity 0
                self._push_down(tier_idx + 1, key, entry, now)
                return
            victim_key, victim = self._pick_eviction_victim(tier_idx, now)
            self._tier_remove(tier_idx, victim_key)
            self._push_down(tier_idx + 1, victim_key, victim, now)
        self._tier_add(tier_idx, key, entry)

    def demote(self):
        """Capacity demotion (Algorithm 1): while a tier exceeds its capacity, its eviction
        victim moves one tier down."""
        with self.lock:
            now = self.clock()
            for tier_idx in range(3):
                tier = self.tiers[tier_idx]
                while len(tier) > self.capacities[tier_idx]:
                    victim_key, victim = self._pick_eviction_victim(tier_idx, now)
                    self._tier_remove(tier_idx, victim_key)
                    self._push_down(tier_idx + 1, victim_key, victim, now)

    def get(self, key: str) -> Optional[CacheEntry]:
        """Direct lookup by key (verbatim repeats); records the access. With the admission
        filter, every lookup also counts as a request for `key`."""
        with self.lock:
            if self.admission_filter:
                self.frequency.record(key)
            for tier_idx, tier in enumerate(self.tiers):
                if key in tier:
                    entry = tier[key]
                    entry.record_access(self.clock())
                    view = entry
                    if self.placement == "hot" and tier_idx > 0:
                        self._pending_promotions.append((entry, tier_idx))
                    return view
            return None

    def maintain(self):
        """Hot placement: applies the promotions of the hits served since the last call, after
        the answers have been returned, so that promotion is not on the lookup path."""
        with self.lock:
            pending, self._pending_promotions = self._pending_promotions, []
            for entry, _ in pending:
                tier_idx = self.tier_of(entry.key)
                if tier_idx is not None and tier_idx > 0 and self.tiers[tier_idx][entry.key] is entry:
                    self._maybe_promote_hot_entry(entry, tier_idx)

    def tier_of(self, key: str) -> Optional[int]:
        """Index of the tier that holds `key`, or None."""
        with self.lock:
            return next((i for i, tier in enumerate(self.tiers) if key in tier), None)


class BaseController(ABC):
    """Interface shared by the strategy controllers."""

    @abstractmethod
    def select_action(self, context: np.ndarray) -> int:
        ...

    @abstractmethod
    def update(self, context: np.ndarray, action: int, reward: float):
        ...


class LinearThompsonSamplingController(BaseController):
    """Linear Thompson sampling with one Bayesian linear-regression model per strategy
    (Algorithm 2). The state is mapped to `feature_dim` dimensions by a fixed Gaussian
    random projection and normalized to unit length; each strategy keeps the precision
    matrix lambda I + sigma^-2 sum z z^T and the vector sigma^-2 sum r z, and the
    parameter is sampled from the exact Gaussian posterior."""

    def __init__(self, context_dim: int, num_actions: int, sigma: float = 1.0, lam: float = 1.0,
                 feature_dim: int = 32, projection_seed: int = 7):
        self.raw_dim = context_dim
        self.d = min(feature_dim, context_dim)
        self.k = num_actions
        self.sigma = sigma
        self.lam = lam
        rng = np.random.default_rng(projection_seed)
        self.projection = rng.normal(0.0, 1.0 / math.sqrt(self.d), size=(context_dim, self.d))
        self.precision = [lam * np.eye(self.d) for _ in range(num_actions)]
        self.b = [np.zeros(self.d) for _ in range(num_actions)]
        self.t = 0

    def _project(self, context: np.ndarray) -> np.ndarray:
        context = np.nan_to_num(context, nan=0.0, posinf=1.0, neginf=-1.0)
        z = context @ self.projection
        norm = np.linalg.norm(z)
        return z / norm if norm > 0 else z

    def select_action(self, context: np.ndarray) -> int:
        z = self._project(context)
        self.t += 1
        sampled = np.empty(self.k)
        for a in range(self.k):
            cov = np.linalg.inv(self.precision[a])
            mu = cov @ self.b[a]
            try:
                theta = np.random.multivariate_normal(mu, cov)
            except np.linalg.LinAlgError:
                theta = mu
            sampled[a] = float(z @ theta)
        return int(np.argmax(sampled))

    def update(self, context: np.ndarray, action: int, reward: float):
        z = self._project(context)
        self.precision[action] += np.outer(z, z) / (self.sigma ** 2)
        self.b[action] += (reward * z) / (self.sigma ** 2)


class PolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_actions))

    def forward(self, x):
        return self.net(x)


class PolicyGradientController(BaseController, nn.Module):
    """REINFORCE controller (Algorithm 3): two-layer policy network, actions sampled
    from the policy with epsilon exploration, exponential moving-average baseline
    b_t = 0.9 b_{t-1} + 0.1 r_t, and step size alpha_t = alpha_0 / (t + 1)."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, num_actions: int = 6,
                 alpha0: float = 1e-3, epsilon0: float = 0.1, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        nn.Module.__init__(self)
        self.policy_net = PolicyNet(input_dim, hidden_dim, num_actions)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1.0)  # lr is set per step
        self.alpha0 = alpha0
        self.t = 0
        self.epsilon = epsilon0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_actions = num_actions
        self.baseline = 0.0
        self.last_grad_norm = float('nan')  # gradient norm before clipping

    def _current_lr(self) -> float:
        return self.alpha0 / (self.t + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy_net(x)

    def select_action(self, context: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.num_actions))
        try:
            with torch.no_grad():
                x = torch.FloatTensor(np.nan_to_num(context, nan=0.0, posinf=1.0, neginf=-1.0)).unsqueeze(0)
                logits = torch.nan_to_num(self.forward(x), nan=0.0, posinf=10.0, neginf=-10.0)
                probs = F.softmax(logits, dim=-1).squeeze(0) + 1e-10
                return int(torch.multinomial(probs / probs.sum(), 1).item())
        except RuntimeError as e:
            logger.warning(f"Action sampling failed ({e}); random action used")
            return int(np.random.randint(0, self.num_actions))

    def update(self, context: np.ndarray, action: int, reward: float):
        """One policy-gradient step per query."""
        self.t += 1
        self.baseline = 0.9 * self.baseline + 0.1 * reward
        advantage = reward - self.baseline

        x = torch.FloatTensor(np.nan_to_num(context, nan=0.0, posinf=1.0, neginf=-1.0)).unsqueeze(0)
        loss = -F.log_softmax(self.forward(x), dim=-1)[0, action] * advantage
        if not torch.isfinite(loss):
            logger.warning("Non-finite policy-gradient loss; update skipped")
            return
        for g in self.optimizer.param_groups:
            g['lr'] = self._current_lr()
        self.optimizer.zero_grad()
        loss.backward()
        self.last_grad_norm = float(torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0))
        self.optimizer.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class FixedStrategyController(BaseController):
    """Always selects the same strategy (fixed-strategy ablations)."""

    def __init__(self, strategy_idx: int):
        self.strategy_idx = strategy_idx

    def select_action(self, context: np.ndarray) -> int:
        return self.strategy_idx

    def update(self, context: np.ndarray, action: int, reward: float):
        pass
