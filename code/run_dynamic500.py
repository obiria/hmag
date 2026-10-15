"""All eight methods on the non-stationary workload (hmag_datasets.dynamic_workload):
four phases of 125 queries (high locality, topic shift, new queries, returning topics),
under the settings of run_main500.py. DYN_ONLY restricts the run to the listed methods
and keeps the stored rows of the others."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")

import logging
import time
from pathlib import Path

import pandas as pd
import torch

torch.set_num_threads(8)

OUT = Path(os.environ.get("DYN_OUT", "./revision_results/dynamic500"))
OUT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    handlers=[logging.FileHandler(OUT / "run.log"), logging.StreamHandler()])
for noisy in ("httpx", "hmag_system", "hmag_core", "hmag_retrieval", "hmag_verifier", "hmag_baselines",
              "sentence_transformers", "faiss", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("dynamic500")

from hmag_system import HMAGConfig, HMAGSystem  # noqa: E402
from hmag_core import QueryContext  # noqa: E402
from hmag_retrieval import HierarchicalRetriever, OllamaGenerator  # noqa: E402
from hmag_baselines import build_all_baselines, CAGBaseline  # noqa: E402
from hmag_datasets import dynamic_workload  # noqa: E402
from run_hmag500 import wait_for_fast_gpu  # noqa: E402

MODEL = "llama3.2:latest"
METHODS = ["Static RAG", "SCALM", "ContextCache", "CAG", "MinCache", "SmartCache", "Adaptive-RAG", "HMAG"]
ONLY = [m for m in os.environ.get("DYN_ONLY", "").split(",") if m]  # rerun only these methods, keep the other rows
QPP = int(os.environ.get("DYN_QPP", "125"))


def main():
    wl = dynamic_workload(queries_per_phase=QPP, seed=42)
    phases = wl.metadata["phase_labels"]
    retriever = HierarchicalRetriever(use_gpu=True)
    retriever.add_documents(wl.corpus)
    generator = OllamaGenerator(model_name=MODEL)
    rows = []
    out = OUT / "dynamic500_raw.csv"
    if ONLY and out.exists():
        rows = pd.read_csv(out).query("method not in @ONLY").to_dict("records")
    for method in METHODS:
        if ONLY and method not in ONLY:
            continue
        retriever._query_embed_cache.clear()
        ms = wait_for_fast_gpu()
        t0 = time.time()
        if method == "HMAG":
            system = HMAGSystem(HMAGConfig(ollama_model=MODEL), retriever=retriever, generator=generator)
            fn = lambda q: system.process_query(q, context=QueryContext(query=q, timestamp=time.time()))
        else:
            system = build_all_baselines(retriever, generator)[method]
            if isinstance(system, CAGBaseline):
                system.preload(wl.corpus)
            fn = system.process_query
        for i, wq in enumerate(wl.queries):
            r = fn(wq.query)
            rows.append({"method": method, "query_index": i, "phase": phases[i],
                         "latency_ms": r.total_latency_ms, "cache_hit": bool(r.cache_hit),
                         "is_repeat": wq.is_repeat, "probe_ms_per_token": ms})
        pd.DataFrame(rows).to_csv(OUT / "dynamic500_raw.csv", index=False)
        g = pd.DataFrame(rows)
        g = g[g.method == method]
        logger.info(f"done {method} in {(time.time() - t0) / 60:.1f} min: lat={g.latency_ms.mean():.0f} "
                    f"hit={g.cache_hit.mean():.3f} by phase: " +
                    ", ".join(f"{p}={h:.2f}" for p, h in g.groupby('phase', sort=False).cache_hit.mean().items()))
    (OUT / "DONE.marker").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))


if __name__ == "__main__":
    main()
