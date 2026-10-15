"""HMAG on the main streams (three datasets, seeds 1 to 3, 500 queries each).

Environment variables: HMAG_MODEL (default llama3.2:latest), HMAG_SEEDS (default 1,2,3),
HMAG_TEMPERATURE (default 0.7) and MAX_MS_PER_TOKEN. Before every stream, a fixed
200-token generation must decode at no more than MAX_MS_PER_TOKEN ms per token; otherwise
the run waits and repeats the check every minute. The measured speed is stored with
each run. The embedding model runs on the GPU and torch uses eight threads."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")

import logging
import subprocess
import time
from pathlib import Path

import pandas as pd
import requests
import torch

torch.set_num_threads(8)

OUT = Path(os.environ.get("HMAG500_OUT", "./revision_results/hmag500"))
OUT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    handlers=[logging.FileHandler(OUT / "run.log"), logging.StreamHandler()])
for noisy in ("httpx", "hmag_system", "hmag_core", "hmag_retrieval", "hmag_verifier",
              "sentence_transformers", "faiss", "urllib3", "hmag_benchmark"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("hmag500")

from hmag_benchmark import BenchmarkRunConfig, run_method_on_workload, get_workload, aggregate_with_ci  # noqa: E402
from hmag_retrieval import HierarchicalRetriever, OllamaGenerator  # noqa: E402

MODEL = os.environ.get("HMAG_MODEL", "llama3.2:latest")
DATASETS = ["squad_v2", "natural_questions", "ms_marco"]
SEEDS = [int(s) for s in os.environ.get("HMAG_SEEDS", "1,2,3").split(",")]
TEMPERATURE = float(os.environ.get("HMAG_TEMPERATURE", "0.7"))  # sampling temperature of the generator
NUM_QUERIES = 500
MAX_MS_PER_TOKEN = float(os.environ.get("MAX_MS_PER_TOKEN", "15"))


def probe_ms_per_token() -> float:
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": MODEL, "prompt": "Write 200 words about the history of Normandy.", "stream": False,
        "options": {"num_predict": 200, "temperature": 0}}, timeout=300).json()
    return r["eval_duration"] / 1e6 / r["eval_count"]


def gpu_state() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=pstate,clocks.sm,power.draw", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "n/a"


def wait_for_fast_gpu() -> float:
    while True:
        ms = probe_ms_per_token()
        if ms <= MAX_MS_PER_TOKEN:
            return ms
        logger.warning(f"GPU slow: {ms:.1f} ms/token (> {MAX_MS_PER_TOKEN}), state {gpu_state()}; waiting 60s")
        time.sleep(60)


def main():
    run_cfg = BenchmarkRunConfig(llm_models=[MODEL], num_queries=NUM_QUERIES, seeds=SEEDS, datasets=DATASETS,
                                 methods=["HMAG"], result_path=str(OUT), enable_profiling=True,
                                 enable_concurrency_test=False)
    run_cfg.ollama_model = MODEL
    tag = MODEL.replace(":", "_").replace(".", "") + ("" if TEMPERATURE == 0.7 else f"_temp{TEMPERATURE:g}")
    parts = []
    for seed in SEEDS:
        for ds in DATASETS:
            wl = get_workload(ds, NUM_QUERIES, seed)
            retriever = HierarchicalRetriever(use_gpu=True)
            retriever.add_documents(wl.corpus)
            ms = wait_for_fast_gpu()
            logger.info(f"start {ds} s{seed}: probe {ms:.1f} ms/token, GPU {gpu_state()}")
            t0 = time.time()
            df = run_method_on_workload("HMAG", wl, run_cfg, seed, retriever, OllamaGenerator(model_name=MODEL, temperature=TEMPERATURE))
            df["temperature"] = TEMPERATURE
            df["probe_ms_per_token_before"] = ms
            df["probe_ms_per_token_after"] = probe_ms_per_token()
            parts.append(df)
            pd.concat(parts, ignore_index=True).to_csv(OUT / f"hmag500_raw_{tag}.csv", index=False)
            m = df[~df["is_warmup"]]
            logger.info(f"done {ds} s{seed} in {(time.time() - t0) / 60:.1f} min: lat={m['latency_ms'].mean():.0f}ms "
                        f"hit={m['cache_hit'].mean():.3f} ret={m['used_retrieval'].mean():.3f} "
                        f"rouge={m['rouge_l'].mean():.3f} repeat={m['is_repeat'].mean():.3f} "
                        f"probe_after={df['probe_ms_per_token_after'].iloc[0]:.1f}")
    raw = pd.concat(parts, ignore_index=True)
    aggregate_with_ci(raw).to_csv(OUT / f"hmag500_agg_{tag}.csv", index=False)
    (OUT / "DONE.marker").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))


if __name__ == "__main__":
    main()
