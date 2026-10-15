"""The seven baselines on the main streams, and the concurrency measurement.

  baselines    the seven baselines on every (seed, dataset) stream, under the settings of
               run_hmag500.py (HMAG_MODEL and HMAG_SEEDS select the model and seeds)
  concurrency  the methods in CONC_METHODS (default: all eight) at the client counts in
               CONC_LEVELS (default 1,2,3,4) on the first 100 queries of the SQuAD v2.0
               seed-1 stream, with a fresh system per level

The query-embedding memo of the shared retriever is cleared before every method, so that
every method pays the same encoding cost."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import torch

torch.set_num_threads(8)

OUT = Path(os.environ.get("MAIN500_OUT", "./revision_results/main500"))
OUT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    handlers=[logging.FileHandler(OUT / "run.log"), logging.StreamHandler()])
for noisy in ("httpx", "hmag_system", "hmag_core", "hmag_retrieval", "hmag_verifier", "hmag_baselines",
              "sentence_transformers", "faiss", "urllib3", "hmag_benchmark", "hmag_load_test"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("main500")

from hmag_benchmark import (BenchmarkRunConfig, run_method_on_workload, get_workload,  # noqa: E402
                            _build_system, ALL_METHODS)
from hmag_retrieval import HierarchicalRetriever, OllamaGenerator  # noqa: E402
from hmag_load_test import run_concurrency_sweep  # noqa: E402
from run_hmag500 import wait_for_fast_gpu, probe_ms_per_token  # noqa: E402

MODEL = os.environ.get("HMAG_MODEL", "llama3.2:latest")
DATASETS = ["squad_v2", "natural_questions", "ms_marco"]
SEEDS = [int(s) for s in os.environ.get("HMAG_SEEDS", "1,2,3").split(",")]
NUM_QUERIES = 500
BASELINES = [m for m in ALL_METHODS if m != "HMAG"]


def _cfg():
    run_cfg = BenchmarkRunConfig(llm_models=[MODEL], num_queries=NUM_QUERIES, seeds=SEEDS, datasets=DATASETS,
                                 methods=list(ALL_METHODS), result_path=str(OUT), enable_profiling=True,
                                 enable_concurrency_test=False)
    run_cfg.ollama_model = MODEL
    return run_cfg


def stage_baselines():
    run_cfg = _cfg()
    tag = MODEL.replace(":", "_").replace(".", "")
    raw_path = OUT / f"baselines500_raw_{tag}.csv"
    done = set()
    if raw_path.exists():
        prev = pd.read_csv(raw_path)
        done = set(zip(prev["seed"], prev["dataset"], prev["method"]))
    generator = OllamaGenerator(model_name=MODEL)
    for seed in SEEDS:
        for ds in DATASETS:
            wl = get_workload(ds, NUM_QUERIES, seed)
            retriever = HierarchicalRetriever(use_gpu=True)
            retriever.add_documents(wl.corpus)
            for method in BASELINES:
                if (seed, wl.name, method) in done:
                    continue
                retriever._query_embed_cache.clear()
                ms = wait_for_fast_gpu()
                t0 = time.time()
                try:
                    df = run_method_on_workload(method, wl, run_cfg, seed, retriever, generator)
                except Exception as e:
                    logger.error(f"{method} {ds} s{seed} crashed: {e}")
                    continue
                df["probe_ms_per_token_before"] = ms
                df["probe_ms_per_token_after"] = probe_ms_per_token()
                df.to_csv(raw_path, mode="a", index=False, header=not raw_path.exists())
                m = df[~df["is_warmup"]]
                logger.info(f"done {method} {ds} s{seed} in {(time.time() - t0) / 60:.1f} min: "
                            f"lat={m['latency_ms'].mean():.0f}ms hit={m['cache_hit'].mean():.3f} "
                            f"ret={m['used_retrieval'].mean():.3f} rouge={m['rouge_l'].mean():.3f} "
                            f"probe {ms:.1f}->{df['probe_ms_per_token_after'].iloc[0]:.1f}")


def stage_concurrency(queries_per_level=100):
    levels = tuple(int(c) for c in os.environ.get("CONC_LEVELS", "1,2,3,4").split(","))
    run_cfg = _cfg()
    wl = get_workload("squad_v2", NUM_QUERIES, 1)
    queries = [wq.query for wq in wl.queries]
    retriever = HierarchicalRetriever(use_gpu=True)
    retriever.add_documents(wl.corpus)
    generator = OllamaGenerator(model_name=MODEL)
    methods = [m for m in os.environ.get("CONC_METHODS", ",".join(ALL_METHODS)).split(",") if m]
    tag = MODEL.replace(':', '_').replace('.', '')
    out = OUT / f"concurrency500_{tag}{os.environ.get('CONC_SUFFIX', '')}.csv"
    rows = []
    for method in methods:
        def factory(method=method):
            retriever._query_embed_cache.clear()
            _, process_fn = _build_system(method, wl, run_cfg, retriever=retriever, generator=generator)
            return process_fn

        ms = wait_for_fast_gpu()
        try:
            for r in run_concurrency_sweep(factory, queries, concurrency_levels=levels,
                                           queries_per_level=queries_per_level):
                row = r.__dict__.copy()
                row.update(method=method, probe_ms_per_token=ms)
                rows.append(row)
                logger.info(f"concurrency {method} c={r.concurrency}: {r.throughput_qps:.2f} QPS "
                            f"mean={r.latency_mean_ms:.0f} p95={r.latency_p95_ms:.0f} errors={r.errors}")
        except Exception as e:
            logger.error(f"concurrency {method} crashed: {e}")
        pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    for st in (sys.argv[1:] or ["baselines", "concurrency"]):
        logger.info(f"stage {st} started")
        {"baselines": stage_baselines, "concurrency": stage_concurrency}[st]()
        logger.info(f"stage {st} done")
