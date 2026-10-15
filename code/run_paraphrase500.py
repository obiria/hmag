"""All eight methods on the paraphrase streams (hmag_paraphrase) of SQuAD v2.0, Natural
Questions and MS MARCO, seeds 1 to 3, 500 queries each, under the settings of
run_main500.py.

  bank   build the paraphrase banks (paraphraser llama3:latest)
  bench  run the benchmark; resumes from the raw file if restarted"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import torch

torch.set_num_threads(8)

OUT = Path(os.environ.get("PARA_OUT", "./revision_results/paraphrase500"))
OUT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    handlers=[logging.FileHandler(OUT / "run.log"), logging.StreamHandler()])
for noisy in ("httpx", "hmag_system", "hmag_core", "hmag_retrieval", "hmag_verifier", "hmag_baselines",
              "sentence_transformers", "faiss", "urllib3", "hmag_benchmark"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("paraphrase500")

from hmag_benchmark import BenchmarkRunConfig, run_method_on_workload, ALL_METHODS  # noqa: E402
from hmag_retrieval import HierarchicalRetriever, OllamaGenerator  # noqa: E402
from hmag_paraphrase import build_bank, paraphrase_workload, zipf_paraphrase_workload, PARAPHRASER  # noqa: E402
from run_hmag500 import wait_for_fast_gpu, probe_ms_per_token  # noqa: E402

MODEL = "llama3.2:latest"
DATASETS = ["squad_v2", "natural_questions", "ms_marco"]
SEEDS = [1, 2, 3]
NUM_QUERIES = 500


def stage_bank():
    for ds in DATASETS:
        t0 = time.time()
        p = build_bank(ds)
        logger.info(f"bank {ds} -> {p} ({(time.time() - t0) / 60:.1f} min)")
    # unload the paraphraser so that it does not share GPU memory with the generator
    requests.post("http://localhost:11434/api/generate", json={"model": PARAPHRASER, "keep_alive": 0}, timeout=120)


def stage_bench():
    run_cfg = BenchmarkRunConfig(llm_models=[MODEL], num_queries=NUM_QUERIES, seeds=SEEDS, datasets=DATASETS,
                                 methods=list(ALL_METHODS), result_path=str(OUT), enable_profiling=True,
                                 enable_concurrency_test=False)
    run_cfg.ollama_model = MODEL
    raw_path = OUT / "paraphrase500_raw.csv"
    done = set()
    if raw_path.exists():
        prev = pd.read_csv(raw_path)
        done = set(zip(prev["seed"], prev["dataset"], prev["method"]))
    generator = OllamaGenerator(model_name=MODEL)
    for seed in SEEDS:
        for ds in DATASETS:
            # PARA_WORKLOAD=zipf: skewed popularity with reworded repeats (main comparison)
            wl = (zipf_paraphrase_workload(ds, NUM_QUERIES, seed) if os.environ.get("PARA_WORKLOAD") == "zipf"
                  else paraphrase_workload(ds, NUM_QUERIES, seed))
            retriever = HierarchicalRetriever(use_gpu=True)
            retriever.add_documents(wl.corpus)
            for method in ALL_METHODS:
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
                df["kind"] = wl.metadata["kinds"]
                df["probe_ms_per_token_before"] = ms
                df["probe_ms_per_token_after"] = probe_ms_per_token()
                df.to_csv(raw_path, mode="a", index=False, header=not raw_path.exists())
                m = df[~df["is_warmup"]]
                par = m[m["kind"] == "paraphrase"]
                logger.info(f"done {method} {ds} s{seed} in {(time.time() - t0) / 60:.1f} min: "
                            f"lat={m['latency_ms'].mean():.0f} hit={m['cache_hit'].mean():.3f} "
                            f"paraphrase_hit={par['cache_hit'].mean():.3f} rouge={m['rouge_l'].mean():.3f}")


if __name__ == "__main__":
    for st in (sys.argv[1:] or ["bank", "bench"]):
        logger.info(f"stage {st} started")
        {"bank": stage_bank, "bench": stage_bench}[st]()
        logger.info(f"stage {st} done")
