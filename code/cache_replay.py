"""Trace replay of HMAG's answer cache without generation.

Whether a query is a cache hit depends only on the query stream, the query embeddings and
the cache policy, not on the generated answers. This script replays a workload through
HierarchicalCache with the request sequence of HMAGSystem.process_query: an exact lookup,
then one semantic search over stored answers, then storage of the answer under the
question as asked. A virtual clock advances by the recorded (or a constant) latency per
query, so the time-dependent priority terms see the same time scale as a real run.

Usage:
  python cache_replay.py validate            # compare with the recorded capacity-pressure runs
  python cache_replay.py compare SEEDS...    # policies on capacity-pressure streams
Output: revision_results/replay/."""

import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from hmag_core import HierarchicalCache
from hmag_datasets import load_ms_marco, load_natural_questions, load_squad_v2
from hmag_system import HMAGConfig

OUT = Path("revision_results/replay")
OUT.mkdir(parents=True, exist_ok=True)
LOADERS = {"squad_v2": load_squad_v2, "natural_questions": load_natural_questions, "ms_marco": load_ms_marco}
STRESS = dict(num_queries=2000, alpha=0.8)
STRESS_TIERS = (10, 30, 60)
# Configurations of the capacity-pressure experiment (run_stress.py).
CONFIGS = {
    "full_system": dict(tiers=STRESS_TIERS, cache_policy="learned"),
    "hierarchical_lru": dict(tiers=STRESS_TIERS, cache_policy="lru"),
    "hierarchical_lfu": dict(tiers=STRESS_TIERS, cache_policy="lfu"),
    "hmag_fixed_temp": dict(tiers=STRESS_TIERS, cache_policy="fixed_temp"),
    "no_hierarchical_cache": dict(tiers=(100, 0, 0), cache_policy="learned"),
}


class VirtualClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def embeddings_for(queries: Sequence[str], tag: str) -> Dict[str, np.ndarray]:
    """Raw embeddings of the distinct queries, encoded one at a time as the system does,
    and stored under revision_results/replay/emb_<tag>.npz."""
    path = OUT / f"emb_{tag}.npz"
    distinct = list(dict.fromkeys(queries))
    if path.exists():
        data = np.load(path, allow_pickle=True)
        stored = dict(zip(data["texts"].tolist(), data["emb"]))
        if all(q in stored for q in distinct):
            return stored
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(HMAGConfig().retrieval_model, device="cuda")
    emb = np.stack([encoder.encode(q, convert_to_numpy=True) for q in distinct])
    np.savez(path, texts=np.array(distinct, dtype=object), emb=emb)
    return dict(zip(distinct, emb))


def replay(queries: Sequence[str], raw_emb: Dict[str, np.ndarray], tiers=STRESS_TIERS,
           latencies_s: Optional[Sequence[float]] = None, constant_latency_s: float = 0.33,
           _out: Optional[list] = None, **cache_kwargs) -> np.ndarray:
    """Returns a boolean hit vector for the stream. `latencies_s` gives the recorded
    duration of every query; otherwise every query takes `constant_latency_s`."""
    cfg = HMAGConfig()
    clock = VirtualClock()
    cache = HierarchicalCache(
        l1_size=tiers[0], l2_size=tiers[1], l3_size=tiers[2], embedding_dim=768,
        similarity_thresholds=(cfg.similarity_threshold,) * 3, n_min=cfg.cache_n_min,
        priority_weights=cfg.priority_weights, promote_threshold=cfg.promote_threshold,
        demote_threshold=cfg.demote_threshold, diversity_lambda=cfg.diversity_lambda,
        encoder=object(), clock=clock, **cache_kwargs)
    norm = {q: HierarchicalCache._normalize(e) for q, e in raw_emb.items()}
    cache.encode_cached = lambda text: norm[text]
    if _out is not None:
        _out.append(cache)

    hits = np.zeros(len(queries), dtype=bool)
    for i, q in enumerate(queries):
        duration = latencies_s[i] if latencies_s is not None else constant_latency_s
        key = hashlib.sha256(q.encode()).hexdigest()
        entry = cache.get(key)
        if entry is not None and entry.entry_type == "qa_response":
            hits[i] = True
        else:
            _, hits[i], _ = cache.search(q, k=cfg.retrieval_top_k, entry_type_filter="qa_response")
            clock.now += duration  # the answer is stored after generation
            cache.put(key=key, content="", embedding_text=q, entry_type="qa_response")
            continue
        clock.now += duration
    return hits


def stream(dataset: str, seed: int):
    wl = LOADERS[dataset](seed=seed, **STRESS)
    return [w.query for w in wl.queries], np.array([w.is_warmup for w in wl.queries])


def validate():
    """Replays the recorded seed-1 capacity-pressure runs with their recorded latencies
    and compares the hit vectors with the recorded ones."""
    rows = []
    for ds in LOADERS:
        queries, warm = stream(ds, 1)
        emb = embeddings_for(queries, f"{ds}_s1")
        rec = pd.read_csv(f"revision_results/stress/stress_raw_{ds}.csv",
                          usecols=["config", "query_index", "latency_ms", "cache_hit"])
        for name, c in CONFIGS.items():
            r = rec[rec.config == name].sort_values("query_index")
            if len(r) != len(queries):
                continue
            sim = replay(queries, emb, tiers=c["tiers"], latencies_s=r.latency_ms.values / 1000,
                         cache_policy=c["cache_policy"])
            real = r.cache_hit.values.astype(bool)
            rows.append(dict(dataset=ds, config=name, recorded=real[~warm].mean(), replayed=sim[~warm].mean(),
                             agreement=(sim == real).mean()))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "validation.csv", index=False)


POLICIES = {
    "HMAG": dict(tiers=STRESS_TIERS, cache_policy="learned"),
    "HMAG + admission": dict(tiers=STRESS_TIERS, cache_policy="learned", admission_filter=True),
    "Hierarchical-LFU": dict(tiers=STRESS_TIERS, cache_policy="lfu"),
    "Hierarchical-LRU": dict(tiers=STRESS_TIERS, cache_policy="lru"),
    "Adaptive-Flat": dict(tiers=(100, 0, 0), cache_policy="learned"),
}


def compare(seeds: List[int]):
    rows = []
    for seed in seeds:
        for ds in LOADERS:
            queries, warm = stream(ds, seed)
            emb = embeddings_for(queries, f"{ds}_s{seed}")
            hit_vectors = {}
            for name, p in POLICIES.items():
                p = dict(p)
                hit_vectors[name] = replay(queries, emb, tiers=p.pop("tiers"), **p)[~warm]
            for name, h in hit_vectors.items():
                rows.append(dict(seed=seed, dataset=ds, policy=name, hit_rate=h.mean(),
                                 only_this=int((h & ~hit_vectors["HMAG"]).sum()),
                                 only_hmag=int((~h & hit_vectors["HMAG"]).sum())))
            print(pd.DataFrame(rows[-len(POLICIES):]).to_string(index=False), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"compare_s{'_'.join(map(str, seeds))}.csv", index=False)
    print(df.pivot_table(index="policy", columns="dataset", values="hit_rate").round(4).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "validate":
        validate()
    elif sys.argv[1] == "compare":
        compare([int(s) for s in sys.argv[2:]])
