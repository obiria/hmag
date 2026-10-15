"""ContextCache and SmartCache on the main, concurrency, dynamic and paraphrase streams,
with every question treated as its own single-turn session.
Output: revision_results/ctxfix/; build_v3_tables.py uses these rows for both methods.
Stages: main concurrency dynamic paraphrase."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import torch

torch.set_num_threads(8)

OUT = Path("./revision_results/ctxfix")
OUT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    handlers=[logging.FileHandler(OUT / "run.log"), logging.StreamHandler()])
for noisy in ("httpx", "hmag_system", "hmag_core", "hmag_retrieval", "hmag_verifier", "hmag_baselines",
              "sentence_transformers", "faiss", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("ctxfix")
METHODS = ["ContextCache", "SmartCache"]

from hmag_baselines import build_all_baselines  # noqa: E402
from hmag_benchmark import BenchmarkRunConfig, get_workload, run_method_on_workload  # noqa: E402
from hmag_datasets import dynamic_workload  # noqa: E402
from hmag_paraphrase import paraphrase_workload  # noqa: E402
from hmag_retrieval import HierarchicalRetriever, OllamaGenerator  # noqa: E402
from run_hmag500 import probe_ms_per_token, wait_for_fast_gpu  # noqa: E402

MODEL = "llama3.2:latest"
DATASETS = ["squad_v2", "natural_questions", "ms_marco"]
SEEDS = [1, 2, 3]
NUM_QUERIES = 500


def _cfg():
    c = BenchmarkRunConfig(llm_models=[MODEL], num_queries=NUM_QUERIES, seeds=SEEDS, datasets=DATASETS,
                           methods=list(METHODS), result_path=str(OUT), enable_profiling=True,
                           enable_concurrency_test=False)
    c.ollama_model = MODEL
    return c


def _bench(workload_fn, name, kinds=False):
    run_cfg = _cfg()
    raw_path = OUT / f"ctx_{name}_raw.csv"
    done = set()
    if raw_path.exists():
        prev = pd.read_csv(raw_path)
        done = set(zip(prev["seed"], prev["dataset"], prev["method"]))
    generator = OllamaGenerator(model_name=MODEL)
    for seed in SEEDS:
        for ds in DATASETS:
            wl = workload_fn(ds, NUM_QUERIES, seed)
            retriever = HierarchicalRetriever(use_gpu=True)
            retriever.add_documents(wl.corpus)
            for method in METHODS:
                if (seed, wl.name, method) in done:
                    continue
                retriever._query_embed_cache.clear()
                ms = wait_for_fast_gpu()
                t0 = time.time()
                df = run_method_on_workload(method, wl, run_cfg, seed, retriever, generator)
                if kinds:
                    df["kind"] = wl.metadata["kinds"]
                df["probe_ms_per_token_before"] = ms
                df["probe_ms_per_token_after"] = probe_ms_per_token()
                df.to_csv(raw_path, mode="a", index=False, header=not raw_path.exists())
                m = df[~df["is_warmup"]]
                logger.info(f"done {method} {name} {ds} s{seed} in {(time.time() - t0) / 60:.1f} min: "
                            f"lat={m['latency_ms'].mean():.0f} hit={m['cache_hit'].mean():.3f} "
                            f"rouge={m['rouge_l'].mean():.3f}")


def stage_main():
    _bench(get_workload, "main")


def stage_paraphrase():
    _bench(paraphrase_workload, "paraphrase", kinds=True)


def stage_concurrency():
    os.environ["CONC_METHODS"] = ",".join(METHODS)
    os.environ["CONC_SUFFIX"] = "_ctxfix"
    import run_main500
    run_main500.OUT = OUT
    run_main500.stage_concurrency()


def stage_dynamic():
    wl = dynamic_workload(queries_per_phase=125, seed=42)
    phases = wl.metadata["phase_labels"]
    retriever = HierarchicalRetriever(use_gpu=True)
    retriever.add_documents(wl.corpus)
    generator = OllamaGenerator(model_name=MODEL)
    rows = []
    for method in METHODS:
        retriever._query_embed_cache.clear()
        system = build_all_baselines(retriever, generator)[method]
        ms = wait_for_fast_gpu()
        for i, wq in enumerate(wl.queries):
            r = system.process_query(wq.query)
            rows.append({"method": method, "query_index": i, "phase": phases[i], "latency_ms": r.total_latency_ms,
                         "cache_hit": bool(r.cache_hit), "is_repeat": wq.is_repeat, "probe_ms_per_token": ms})
        g = pd.DataFrame(rows); g = g[g.method == method]
        logger.info(f"done {method} dynamic: lat={g.latency_ms.mean():.0f} hit={g.cache_hit.mean():.3f}")
    pd.DataFrame(rows).to_csv(OUT / "ctx_dynamic_raw.csv", index=False)


if __name__ == "__main__":
    for st in (sys.argv[1:] or ["main", "concurrency", "dynamic", "paraphrase"]):
        logger.info(f"stage {st} started")
        {"main": stage_main, "concurrency": stage_concurrency,
         "dynamic": stage_dynamic, "paraphrase": stage_paraphrase}[st]()
        logger.info(f"stage {st} done")
    (OUT / "DONE.marker").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
