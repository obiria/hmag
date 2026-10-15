"""Query workloads built from SQuAD v2.0, Natural Questions and MS MARCO, and the
synthetic non-stationary workload.

Each loader returns a `RAGWorkload`: the retrieval corpus and an ordered query stream
with warm-up flags, repetition flags and reference answers. Streams are drawn with Zipf
popularity and are reproducible from the seed."""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from hmag_retrieval import Document

logger = logging.getLogger(__name__)


@dataclass
class WorkloadQuery:
    query: str
    reference_answer: Optional[str]
    gold_doc_ids: List[str]
    is_warmup: bool
    is_repeat: bool  # the same question text occurred earlier in the stream


@dataclass
class RAGWorkload:
    name: str
    corpus: List[Document]
    queries: List[WorkloadQuery]
    metadata: Dict = field(default_factory=dict)


def _doc_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _power_law_order(pool_size: int, num_queries: int, alpha: float, seed: int) -> np.ndarray:
    """Draws `num_queries` indices from a pool of `pool_size` questions with Zipf
    popularity P(q_i) proportional to i^-alpha."""
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, pool_size + 1, dtype=np.float64)
    weights = ranks ** (-alpha)
    probs = weights / weights.sum()
    return rng.choice(pool_size, size=num_queries, p=probs)


def _finalize_stream(raw_queries: List[WorkloadQuery], num_queries: int, alpha: float,
                      warmup_fraction: float, seed: int) -> List[WorkloadQuery]:
    if not raw_queries:
        return []
    indices = _power_law_order(len(raw_queries), num_queries, alpha, seed)
    seen = set()
    warmup_count = int(num_queries * warmup_fraction)
    stream = []
    for i, idx in enumerate(indices):
        base = raw_queries[int(idx)]
        text = base.query
        stream.append(WorkloadQuery(
            query=text,
            reference_answer=base.reference_answer,
            gold_doc_ids=base.gold_doc_ids,
            is_warmup=(i < warmup_count),
            is_repeat=(text in seen),
        ))
        seen.add(text)
    return stream


def load_squad_v2(num_queries: int = 500, corpus_articles: int = 200,
                   alpha: float = 1.2, warmup_fraction: float = 0.1,
                   seed: int = 42) -> Optional[RAGWorkload]:
    """SQuAD v2.0, validation split. Corpus: the paragraphs of the first `corpus_articles`
    articles; questions: those written against these paragraphs."""
    try:
        from datasets import load_dataset
        ds = load_dataset("rajpurkar/squad_v2", split="validation")
    except Exception as e:
        logger.warning(f"Could not load SQuAD v2.0: {e}")
        return None

    corpus_map: Dict[str, Document] = {}
    raw_queries: List[WorkloadQuery] = []
    seen_titles = set()

    for ex in ds:
        if len(seen_titles) >= corpus_articles and ex['title'] not in seen_titles:
            continue
        seen_titles.add(ex['title'])
        did = _doc_id(ex['context'])
        if did not in corpus_map:
            corpus_map[did] = Document(doc_id=did, content=ex['context'], title=ex['title'],
                                        metadata={'dataset': 'squad_v2'})
        answer = ex['answers']['text'][0] if ex['answers']['text'] else None
        raw_queries.append(WorkloadQuery(ex['question'], answer, [did], False, False))
        if len(raw_queries) >= max(num_queries * 5, 1000):
            break

    workload_queries = _finalize_stream(raw_queries, num_queries, alpha, warmup_fraction, seed)
    return RAGWorkload(
        name="squad_v2", corpus=list(corpus_map.values()), queries=workload_queries,
        metadata={'source_articles': len(seen_titles), 'raw_query_pool': len(raw_queries),
                  'alpha': alpha, 'warmup_fraction': warmup_fraction, 'seed': seed},
    )


def _local_cached_file(repo_id: str, filename: str) -> Optional[str]:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    p = try_to_load_from_cache(repo_id, filename, repo_type="dataset")
    return p if isinstance(p, str) else None


def _local_nq_validation_shards() -> Optional[List[str]]:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    paths = []
    for i in range(7):
        p = try_to_load_from_cache("google-research-datasets/natural_questions",
                                   f"default/validation-0000{i}-of-00007.parquet", repo_type="dataset")
        if not isinstance(p, str):
            return None
        paths.append(p)
    return paths


def load_natural_questions(num_queries: int = 500, corpus_docs: int = 200,
                            alpha: float = 1.2, warmup_fraction: float = 0.1,
                            seed: int = 42) -> Optional[RAGWorkload]:
    """Natural Questions, validation split. Corpus: the long-answer passages of the first
    examples (at most `corpus_docs`); questions: the corresponding search queries."""
    try:
        from datasets import load_dataset
        local = _local_nq_validation_shards()
        if local:
            # the seven validation shards of the "default" configuration, from the local cache
            ds = load_dataset("parquet", data_files={"validation": local}, split="validation", streaming=True)
        else:
            ds = load_dataset("google-research-datasets/natural_questions", "default",
                               split="validation", streaming=True)
    except Exception as e:
        logger.warning(f"Could not load Natural Questions: {e}")
        return None

    corpus_map: Dict[str, Document] = {}
    raw_queries: List[WorkloadQuery] = []

    try:
        for ex in ds:
            tokens = ex['document']['tokens']['token']
            la = ex['annotations']['long_answer'][0] if ex['annotations']['long_answer'] else None
            if not la or la['start_token'] < 0 or la['end_token'] <= la['start_token']:
                continue
            passage = " ".join(tokens[la['start_token']:la['end_token']])[:4000]
            if not passage.strip():
                continue
            did = _doc_id(passage)
            if did not in corpus_map and len(corpus_map) < corpus_docs:
                corpus_map[did] = Document(doc_id=did, content=passage,
                                            title=ex['document'].get('title', ''),
                                            metadata={'dataset': 'natural_questions'})
            short_answers = ex['annotations']['short_answers'][0]['text'] if ex['annotations']['short_answers'] else []
            answer = short_answers[0] if short_answers else None
            raw_queries.append(WorkloadQuery(ex['question']['text'], answer, [did], False, False))
            if len(raw_queries) >= max(num_queries * 5, 1000):
                break
    except Exception as e:
        logger.warning(f"Error streaming Natural Questions after {len(raw_queries)} examples: {e}")

    if not raw_queries:
        return None

    workload_queries = _finalize_stream(raw_queries, num_queries, alpha, warmup_fraction, seed)
    return RAGWorkload(
        name="natural_questions", corpus=list(corpus_map.values()), queries=workload_queries,
        metadata={'source_docs': len(corpus_map), 'raw_query_pool': len(raw_queries),
                  'alpha': alpha, 'warmup_fraction': warmup_fraction, 'seed': seed},
    )


def load_ms_marco(num_queries: int = 500, corpus_size: int = 2000,
                   alpha: float = 1.2, warmup_fraction: float = 0.1,
                   seed: int = 42) -> Optional[RAGWorkload]:
    """MS MARCO v2.1, validation split. Corpus: the selected passages of the questions
    (at most `corpus_size`); questions: the corresponding search queries."""
    try:
        from datasets import load_dataset
        local = _local_cached_file("microsoft/ms_marco", "v2.1/validation-00000-of-00001.parquet")
        if local:
            # the v2.1 validation split, from the local cache
            ds = load_dataset("parquet", data_files={"validation": [local]}, split="validation", streaming=True)
        else:
            ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation", streaming=True)
    except Exception as e:
        logger.warning(f"Could not load MS MARCO: {e}")
        return None

    corpus_map: Dict[str, Document] = {}
    raw_queries: List[WorkloadQuery] = []

    try:
        for ex in ds:
            passages = ex['passages']['passage_text']
            selected = [p for p, sel in zip(passages, ex['passages']['is_selected']) if sel == 1]
            gold_ids = []
            for p in (selected or passages[:1]):
                did = _doc_id(p)
                if did not in corpus_map and len(corpus_map) < corpus_size:
                    corpus_map[did] = Document(doc_id=did, content=p, metadata={'dataset': 'ms_marco'})
                gold_ids.append(did)
            answer = ex['answers'][0] if ex.get('answers') else None
            raw_queries.append(WorkloadQuery(ex['query'], answer, gold_ids, False, False))
            if len(raw_queries) >= max(num_queries * 5, 1000):
                break
    except Exception as e:
        logger.warning(f"Error streaming MS MARCO after {len(raw_queries)} examples: {e}")

    if not raw_queries:
        return None

    workload_queries = _finalize_stream(raw_queries, num_queries, alpha, warmup_fraction, seed)
    return RAGWorkload(
        name="ms_marco", corpus=list(corpus_map.values()), queries=workload_queries,
        metadata={'source_passages': len(corpus_map), 'raw_query_pool': len(raw_queries),
                  'alpha': alpha, 'warmup_fraction': warmup_fraction, 'seed': seed},
    )


LOADERS = {
    "squad_v2": load_squad_v2,
    "natural_questions": load_natural_questions,
    "ms_marco": load_ms_marco,
}


def dynamic_workload(queries_per_phase: int = 125, seed: int = 42) -> RAGWorkload:
    """Non-stationary workload: four consecutive phases over three disjoint topic pools.

      1. high locality: pool A, alpha = 2.0
      2. topic shift: pool B, alpha = 2.0
      3. new queries: pool C, alpha = 0.1
      4. returning topics: pool A, alpha = 2.0

    The phase of each query is given in `metadata['phase_labels']`."""
    rng_seed = seed
    pool_a_topics = ["artificial intelligence", "quantum computing", "climate change", "space exploration"]
    pool_b_topics = ["biotechnology", "renewable energy", "blockchain", "neuroscience"]
    pool_c_topics = ["nanotechnology", "genomics", "robotics", "cybersecurity"]
    templates = ["What is", "How does", "Explain", "Compare", "Why is", "What are the benefits of"]

    def _pool(topics: List[str]) -> List[str]:
        return [f"{t} {topic}?" for t in templates for topic in topics]

    def _phase_queries(pool: List[str], alpha: float, n: int, seed: int) -> List[WorkloadQuery]:
        indices = _power_law_order(len(pool), n, alpha, seed)
        seen = set()
        out = []
        for idx in indices:
            text = pool[int(idx)]
            out.append(WorkloadQuery(text, None, [], False, text in seen))
            seen.add(text)
        return out

    pool_a, pool_b, pool_c = _pool(pool_a_topics), _pool(pool_b_topics), _pool(pool_c_topics)
    phases = [
        ("high_locality", pool_a, 2.0),
        ("topic_shift", pool_b, 2.0),
        ("novel_queries", pool_c, 0.1),
        ("returning_topics", pool_a, 2.0),
    ]

    queries: List[WorkloadQuery] = []
    phase_labels: List[str] = []
    for i, (label, pool, alpha) in enumerate(phases):
        phase_qs = _phase_queries(pool, alpha, queries_per_phase, rng_seed + i)
        queries.extend(phase_qs)
        phase_labels.extend([label] * len(phase_qs))

    all_topics = pool_a_topics + pool_b_topics + pool_c_topics
    corpus = [
        Document(doc_id=f"dynamic_doc_{i}",
                  content=f"This is a reference document about {topic}, covering key concepts, "
                          f"applications, and open challenges.",
                  metadata={'dataset': 'dynamic', 'topic': topic})
        for i, topic in enumerate(all_topics)
    ]

    return RAGWorkload(
        name="dynamic_workload", corpus=corpus, queries=queries,
        metadata={'seed': seed, 'queries_per_phase': queries_per_phase, 'phase_labels': phase_labels},
    )
