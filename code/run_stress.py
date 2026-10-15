"""Capacity-pressure experiment (design in STRESS_TEST_DESIGN.md): 2,000-query streams
(seed 1, Zipf alpha = 0.8) on the three datasets with room for 100 answers in every
configuration. HMAG and its variants run through run_revision_experiments._run_stream,
SCALM and MinCache through an equivalent per-query loop. STRESS_ONLY restricts the run to
the listed configurations and keeps the stored rows of the others.
STRESS_SEEDS lists the stream seeds (default 1).
Output: revision_results/stress/stress_raw_<dataset>.csv for seed 1 and
stress_raw_<dataset>_s<seed>.csv for the other seeds."""
import json
import os
import time
from pathlib import Path

import pandas as pd

os.environ.setdefault("REV_OUT", "revision_results/stress")
import run_revision_experiments as R  # noqa: E402
from hmag_baselines import MinCacheBaseline, SCALMBaseline  # noqa: E402
from hmag_datasets import load_ms_marco, load_natural_questions, load_squad_v2  # noqa: E402
from hmag_qa_metrics import score as qa_score  # noqa: E402
from hmag_retrieval import HierarchicalRetriever  # noqa: E402
from hmag_system import HMAGConfig, HMAGSystem  # noqa: E402

OUT = Path(os.environ.get("STRESS_OUT", "revision_results/stress"))
OUT.mkdir(parents=True, exist_ok=True)
LOADERS = {"squad_v2": load_squad_v2, "natural_questions": load_natural_questions, "ms_marco": load_ms_marco}
N_QUERIES, ALPHA = 2000, 0.8
SEEDS = [int(s) for s in os.environ.get("STRESS_SEEDS", "1").split(",") if s]
TIERS = {"l1_cache_size": 10, "l2_cache_size": 30, "l3_cache_size": 60}
# The replacement-policy variants run without the admission filter.
HMAG_CONFIGS = {
    "full_system": {},
    "hmag_no_admission": {"cache_admission": False},
    "hierarchical_lru": {"cache_policy": "lru", "cache_admission": False},
    "hierarchical_lfu": {"cache_policy": "lfu", "cache_admission": False},
    "hmag_fixed_temp": {"cache_policy": "fixed_temp", "cache_admission": False},
    "no_hierarchical_cache": {"l1_cache_size": 100, "l2_cache_size": 0, "l3_cache_size": 0,
                              "cache_admission": False},
    # HMAG (original placement): new answers in L1, promotion only into free space
    "hmag_original_placement": {"cache_placement": "insert_l1", "cache_frequency_norm": "exp",
                                "promote_threshold": 0.7, "cache_promote_to_top": False,
                                "cache_promote_sweep": True},
}
BASELINES = {"scalm": SCALMBaseline, "mincache": MinCacheBaseline}


def run_baseline(cls, wl, retriever, generator, extra):
    probe = R._wait_for_fast_gpu()
    system = cls(retriever, generator, capacity=100, top_k=10)
    rows = []
    for i, wq in enumerate(wl.queries):
        r = system.process_query(wq.query)
        err = bool(r.metadata.get("generation_error"))
        bd = r.latency_breakdown or {}
        rows.append({**extra, "query_index": i, "is_warmup": wq.is_warmup, "is_repeat": wq.is_repeat,
                     "latency_ms": r.total_latency_ms, "cache_hit": bool(r.cache_hit),
                     "used_retrieval": bool(r.used_retrieval), "strategy": None, "exact_match": False,
                     **qa_score(wq.reference_answer, r.response, err), "response": r.response,
                     "reference": wq.reference_answer, "error": err,
                     "prompt_tokens": r.metadata.get("prompt_tokens", float("nan")),
                     "generated_tokens": r.metadata.get("generated_tokens", float("nan")),
                     "lat_time_to_first_token_ms": bd.get("time_to_first_token", float("nan")),
                     "lat_decoding_ms": bd.get("decoding", float("nan")), "probe_ms_per_token": probe})
    return pd.DataFrame(rows)


ONLY = [c for c in os.environ.get("STRESS_ONLY", "").split(",") if c]  # rerun only these configurations
DATASETS = [d for d in os.environ.get("STRESS_DATASETS", "").split(",") if d] or list(LOADERS)


if __name__ == "__main__":
    shared = R._shared_models(R.PRIMARY_MODEL)
    for SEED in SEEDS:
        for ds in DATASETS:
            loader = LOADERS[ds]
            wl = loader(num_queries=N_QUERIES, alpha=ALPHA, seed=SEED)
            retriever = HierarchicalRetriever(use_gpu=R.USE_GPU)
            retriever.add_documents(wl.corpus)
            parts = []
            out_file = OUT / (f"stress_raw_{ds}.csv" if SEED == 1 else f"stress_raw_{ds}_s{SEED}.csv")
            if ONLY and out_file.exists():
                old = pd.read_csv(out_file)
                parts.append(old[~old.config.isin(ONLY)])
            for name, ov in HMAG_CONFIGS.items():
                if ONLY and name not in ONLY:
                    continue
                t0 = time.perf_counter()
                retriever._query_embed_cache.clear()
                cfg = HMAGConfig(ollama_model=R.PRIMARY_MODEL)
                for k, v in {**TIERS, **ov}.items():
                    setattr(cfg, k, v)
                system = HMAGSystem(cfg, retriever=retriever, **shared)
                df = R._run_stream(system, wl, {"config": name, "dataset": ds, "seed": SEED})
                parts.append(df)
                m = df[~df.is_warmup]
                cache = system.cache
                stats = {"config": name, "dataset": ds, "seed": SEED, "ram_bytes": cache.ram_bytes(),
                         "entries": [len(t) for t in cache.tiers],
                         # request counts of the admission filter (question key -> count)
                         "admission_keys": len(cache.frequency.counts) if cache.admission_filter else 0}
                with open(OUT / "cache_stats.jsonl", "a") as f:
                    f.write(json.dumps(stats) + "\n")
                R.logger.info(f"stress {ds} s{SEED} {name}: lat={m.latency_ms.mean():.0f} hit={m.cache_hit.mean():.3f} "
                              f"f1={m.token_f1.mean():.3f} ({time.perf_counter() - t0:.0f}s)")
                pd.concat(parts, ignore_index=True).to_csv(out_file, index=False)
            for name, cls in BASELINES.items():
                if ONLY and name not in ONLY:
                    continue
                t0 = time.perf_counter()
                retriever._query_embed_cache.clear()
                df = run_baseline(cls, wl, retriever, shared["generator"], {"config": name, "dataset": ds, "seed": SEED})
                parts.append(df)
                m = df[~df.is_warmup]
                R.logger.info(f"stress {ds} s{SEED} {name}: lat={m.latency_ms.mean():.0f} hit={m.cache_hit.mean():.3f} "
                              f"f1={m.token_f1.mean():.3f} ({time.perf_counter() - t0:.0f}s)")
                pd.concat(parts, ignore_index=True).to_csv(out_file, index=False)
    (OUT / "DONE.marker").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
