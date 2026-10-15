"""Benchmark harness shared by the experiment drivers: builds HMAG or a baseline on a
shared retriever and generator, runs it over a query stream and records per-query
latency, stage timings, cache hits, answer quality and resource use."""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from hmag_baselines import CAGBaseline, build_all_baselines
from hmag_core import QueryContext
from hmag_datasets import LOADERS, RAGWorkload
from hmag_profiling import ResourceMonitor, bootstrap_ci
from hmag_qa_metrics import score as qa_score
from hmag_retrieval import HierarchicalRetriever, OllamaGenerator
from hmag_system import HMAGConfig, HMAGSystem

logger = logging.getLogger(__name__)

ALL_METHODS = ["HMAG", "Static RAG", "SCALM", "ContextCache", "CAG", "MinCache", "SmartCache", "Adaptive-RAG"]


@dataclass
class BenchmarkRunConfig:
    llm_models: List[str] = field(default_factory=lambda: ["llama3.2:latest"])
    controller_type: str = "thompson_sampling"
    num_queries: int = 50
    seeds: List[int] = field(default_factory=lambda: [1, 2, 3])
    datasets: List[str] = field(default_factory=lambda: ["squad_v2", "natural_questions", "ms_marco"])
    methods: List[str] = field(default_factory=lambda: list(ALL_METHODS))
    result_path: str = "./benchmark_results"
    enable_profiling: bool = True
    enable_concurrency_test: bool = True
    concurrency_levels: Tuple[int, ...] = (1, 2, 4)
    concurrency_methods: Optional[List[str]] = None  # None: all of `methods`
    ollama_model: str = "llama3.2:latest"



def get_workload(name: str, num_queries: int, seed: int) -> RAGWorkload:
    loader = LOADERS.get(name)
    wl = loader(num_queries=num_queries, seed=seed) if loader else None
    if wl is None or not wl.queries:
        raise RuntimeError(f"Dataset '{name}' could not be loaded; no substitute workload is used.")
    return wl


def _build_system(method: str, workload: RAGWorkload, run_cfg: BenchmarkRunConfig,
                   retriever: Optional[HierarchicalRetriever] = None,
                   generator: Optional[OllamaGenerator] = None):
    """Builds the system under test. The retriever (with the embedded corpus) and the
    generator are normally shared by all methods run on a workload."""
    if retriever is None:
        retriever = HierarchicalRetriever(use_gpu=False)
        if workload.corpus:
            retriever.add_documents(workload.corpus)
    generator = generator or OllamaGenerator(model_name=run_cfg.ollama_model)

    # CACHE_CAPACITY: equal number of answers for every answer cache; HMAG keeps the
    # proportions of its default tiers (1,500 : 2,100 : 7,500).
    capacity = int(os.environ["CACHE_CAPACITY"]) if os.environ.get("CACHE_CAPACITY") else None
    if method == "HMAG":
        cfg = HMAGConfig(ollama_model=run_cfg.ollama_model, controller_type=run_cfg.controller_type)
        if os.environ.get("CACHE_TIERS"):  # explicit L1,L2,L3
            cfg.l1_cache_size, cfg.l2_cache_size, cfg.l3_cache_size = map(int, os.environ["CACHE_TIERS"].split(","))
        elif capacity is not None:
            cfg.l1_cache_size = max(1, round(capacity * 1500 / 11100))
            cfg.l2_cache_size = max(1, round(capacity * 2100 / 11100))
            cfg.l3_cache_size = capacity - cfg.l1_cache_size - cfg.l2_cache_size
        system = HMAGSystem(cfg, retriever=retriever, generator=generator)
        process_fn = lambda q: system.process_query(q, context=QueryContext(query=q, timestamp=time.time()))
    else:
        baselines = build_all_baselines(retriever, generator, capacity=capacity)
        system = baselines[method]
        if isinstance(system, CAGBaseline):
            system.preload(workload.corpus)
        process_fn = lambda q: system.process_query(q)

    return system, process_fn


def run_method_on_workload(method: str, workload: RAGWorkload, run_cfg: BenchmarkRunConfig,
                            seed: int, retriever: Optional[HierarchicalRetriever] = None,
                            generator: Optional[OllamaGenerator] = None) -> pd.DataFrame:
    logger.info(f"Running {method} on {workload.name} (seed={seed}, n={len(workload.queries)})")
    system, process_fn = _build_system(method, workload, run_cfg, retriever, generator)

    # Tier that served each HMAG answer-cache hit.
    served = {'tier': -1}
    stats = getattr(getattr(system, 'cache', None), 'stats', None)
    if stats is not None:
        record = stats.record_search

        def record_tier(hit, tier_hit):
            if hit and tier_hit is not None:
                served['tier'] = tier_hit
            record(hit, tier_hit)
        stats.record_search = record_tier

    monitor = ResourceMonitor() if run_cfg.enable_profiling else None
    if monitor:
        monitor.start()

    # Stage timings; a stage that a query does not execute is recorded as NaN.
    component_keys = ('cache', 'controller', 'retrieval', 'verification',
                       'time_to_first_token', 'decoding', 'generation')

    rows = []
    for wq in workload.queries:
        breakdown: Dict[str, float] = {}
        metadata: Dict = {}
        served['tier'] = -1
        try:
            result = process_fn(wq.query)
            response_text = result.response
            latency_ms = result.total_latency_ms
            cache_hit = bool(getattr(result, 'cache_hit', False))
            breakdown = dict(getattr(result, 'latency_breakdown', {}) or {})
            metadata = dict(getattr(result, 'metadata', {}) or {})
            # For HMAG, retrieval was used if any document came from the corpus.
            if hasattr(result, 'used_retrieval'):
                used_retrieval = bool(result.used_retrieval)
            else:
                used_retrieval = any(d.source_type != "cache" for d in result.documents_used)
            error = bool(metadata.get('generation_error'))
        except Exception as e:
            logger.warning(f"{method} failed on query '{wq.query[:50]}...': {e}")
            response_text, latency_ms, cache_hit, used_retrieval, error = "", float('nan'), False, False, True

        quality = qa_score(wq.reference_answer, response_text, error)

        row = {
            'method': method, 'model': run_cfg.ollama_model, 'dataset': workload.name, 'seed': seed,
            'query': wq.query, 'is_warmup': wq.is_warmup, 'is_repeat': wq.is_repeat,
            'latency_ms': latency_ms, 'cache_hit': cache_hit, 'used_retrieval': used_retrieval,
            **quality, 'response': response_text, 'reference': wq.reference_answer, 'error': error,
            'prompt_tokens': metadata.get('prompt_tokens', float('nan')),
            'generated_tokens': metadata.get('generated_tokens', float('nan')),
            'cache_tier': served['tier'] + 1 if cache_hit and served['tier'] >= 0 else 0,
        }
        for key in component_keys:
            row[f'latency_{key}_ms'] = breakdown.get(key, float('nan'))
        rows.append(row)

    resource_summary = monitor.stop() if monitor else None
    df = pd.DataFrame(rows)
    if resource_summary is not None:
        for field_name in ('avg_gpu_util_pct', 'peak_gpu_memory_mb', 'avg_gpu_power_w',
                           'energy_joules', 'peak_rss_memory_mb'):
            df[field_name] = getattr(resource_summary, field_name)
    return df


def _aggregate_resources_across_seeds(full_group: pd.DataFrame, resource_cols: List[str]) -> Dict[str, float]:
    """Combines the per-run resource measurements of several seeds: maximum for peak
    values, mean for averages, and total energy divided by the total number of queries."""
    if 'seed' in full_group.columns:
        per_seed = full_group.groupby('seed').agg(
            **{col: (col, 'first') for col in resource_cols if col in full_group.columns},
            _n=('seed', 'size'),
        )
    else:
        per_seed = full_group.iloc[[0]].assign(_n=len(full_group))  # single-seed input

    out: Dict[str, float] = {}
    for col in resource_cols:
        if col not in per_seed.columns or not per_seed[col].notna().any():
            continue
        if col.startswith('peak_'):
            out[col] = float(per_seed[col].dropna().max())
        elif col == 'energy_joules':
            out[col] = float(per_seed[col].dropna().sum())
        else:
            out[col] = float(per_seed[col].dropna().mean())
    if 'energy_joules' in out:
        total_queries = float(per_seed['_n'].sum())
        if total_queries > 0:
            out['energy_joules_per_query'] = out['energy_joules'] / total_queries
    return out


def aggregate_with_ci(df: pd.DataFrame, exclude_warmup: bool = True) -> pd.DataFrame:
    """Per (model, method, dataset): means with 95% bootstrap intervals over the measured
    (non-warm-up) queries. Resource values cover the whole run, including warm-up."""
    resource_cols = ['avg_gpu_util_pct', 'peak_gpu_memory_mb', 'avg_gpu_power_w',
                      'energy_joules', 'peak_rss_memory_mb']
    data = df[~df['is_warmup']] if exclude_warmup else df

    rows = []
    for (model, method, dataset), full_group in df.groupby(['model', 'method', 'dataset']):
        group = data[(data['model'] == model) & (data['method'] == method) & (data['dataset'] == dataset)]
        latencies = group['latency_ms'].dropna().tolist()
        lat_mean, lat_lo, lat_hi = bootstrap_ci(latencies) if latencies else (float('nan'),) * 3
        hit_rate = float(group['cache_hit'].mean()) if len(group) else float('nan')
        retrieval_rate = (float(group['used_retrieval'].mean())
                           if 'used_retrieval' in group and len(group) else float('nan'))
        quality_vals = group['rouge_l'].dropna().tolist()
        q_mean, q_lo, q_hi = bootstrap_ci(quality_vals) if quality_vals else (float('nan'),) * 3

        row = {
            'model': model, 'method': method, 'dataset': dataset, 'n': len(group),
            'latency_ms_mean': lat_mean, 'latency_ms_ci_lo': lat_lo, 'latency_ms_ci_hi': lat_hi,
            'cache_hit_rate': hit_rate, 'retrieval_rate': retrieval_rate,
            'rouge_l_mean': q_mean, 'rouge_l_ci_lo': q_lo, 'rouge_l_ci_hi': q_hi,
            'error_rate': float(group['error'].mean()) if len(group) else float('nan'),
        }

        # mean stage timings and token counts
        for key in ('cache', 'controller', 'retrieval', 'verification',
                    'time_to_first_token', 'decoding', 'generation'):
            col = f'latency_{key}_ms'
            if col in group.columns:
                row[f'{key}_ms_mean'] = float(group[col].dropna().mean()) if group[col].notna().any() else float('nan')
        for col in ('prompt_tokens', 'generated_tokens'):
            if col in group.columns:
                row[f'{col}_mean'] = float(group[col].dropna().mean()) if group[col].notna().any() else float('nan')

        row.update(_aggregate_resources_across_seeds(full_group, resource_cols))

        rows.append(row)
    return pd.DataFrame(rows)
