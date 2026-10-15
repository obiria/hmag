"""CAG baseline on the main, concurrency, dynamic and paraphrase streams, with the
preloaded documents limited to the context window (hmag_baselines.CAGBaseline.preload).
Output: revision_results/cagfix/; build_v3_tables.py uses these rows for CAG.
Stages: smoke main concurrency dynamic paraphrase."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import torch

torch.set_num_threads(8)

OUT = Path("./revision_results/cagfix")
OUT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    handlers=[logging.FileHandler(OUT / "run.log"), logging.StreamHandler()])
for noisy in ("httpx", "hmag_system", "hmag_core", "hmag_retrieval", "hmag_verifier", "hmag_baselines",
              "sentence_transformers", "faiss", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("cagfix")

from hmag_baselines import CAGBaseline  # noqa: E402
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
                           methods=["CAG"], result_path=str(OUT), enable_profiling=True,
                           enable_concurrency_test=False)
    c.ollama_model = MODEL
    return c


def stage_smoke():
    """Every CAG prompt must fit the 4,096-token window on all three corpora."""
    generator = OllamaGenerator(model_name=MODEL)
    for ds in DATASETS:
        wl = get_workload(ds, NUM_QUERIES, 1)
        cag = CAGBaseline(None, generator)
        cag.preload(wl.corpus)
        toks = []
        for wq in wl.queries[:5]:
            r = cag.process_query(wq.query)
            toks.append(r.metadata.get("prompt_tokens"))
        logger.info(f"smoke {ds}: {len(cag._preloaded)} docs preloaded, prompt tokens {toks}, "
                    f"last answer {r.response[:80]!r}")
        if max(toks) >= 4096:
            raise SystemExit(f"CAG prompt still fills the context window on {ds}")


def _bench(workload_fn, name, kinds=False):
    run_cfg = _cfg()
    raw_path = OUT / f"cag_{name}_raw.csv"
    done = set()
    if raw_path.exists():
        prev = pd.read_csv(raw_path)
        done = set(zip(prev["seed"], prev["dataset"]))
    generator = OllamaGenerator(model_name=MODEL)
    for seed in SEEDS:
        for ds in DATASETS:
            wl = workload_fn(ds, NUM_QUERIES, seed)
            if (seed, wl.name) in done:
                continue
            retriever = HierarchicalRetriever(use_gpu=True)
            retriever.add_documents(wl.corpus)
            ms = wait_for_fast_gpu()
            t0 = time.time()
            df = run_method_on_workload("CAG", wl, run_cfg, seed, retriever, generator)
            if kinds:
                df["kind"] = wl.metadata["kinds"]
            df["probe_ms_per_token_before"] = ms
            df["probe_ms_per_token_after"] = probe_ms_per_token()
            df.to_csv(raw_path, mode="a", index=False, header=not raw_path.exists())
            m = df[~df["is_warmup"]]
            logger.info(f"done CAG {name} {ds} s{seed} in {(time.time() - t0) / 60:.1f} min: "
                        f"lat={m['latency_ms'].mean():.0f} rouge={m['rouge_l'].mean():.3f} "
                        f"prompt_tokens={m['prompt_tokens'].mean():.0f}")


def stage_main():
    _bench(get_workload, "main")


def stage_paraphrase():
    _bench(paraphrase_workload, "paraphrase", kinds=True)


def stage_concurrency():
    os.environ["CONC_METHODS"] = "CAG"
    os.environ["CONC_SUFFIX"] = "_cagfix"
    import run_main500
    run_main500.OUT = OUT
    run_main500.stage_concurrency()


def stage_dynamic():
    wl = dynamic_workload(queries_per_phase=125, seed=42)
    phases = wl.metadata["phase_labels"]
    retriever = HierarchicalRetriever(use_gpu=True)
    retriever.add_documents(wl.corpus)
    generator = OllamaGenerator(model_name=MODEL)
    system = CAGBaseline(retriever, generator)
    system.preload(wl.corpus)
    ms = wait_for_fast_gpu()
    rows = []
    for i, wq in enumerate(wl.queries):
        r = system.process_query(wq.query)
        rows.append({"method": "CAG", "query_index": i, "phase": phases[i], "latency_ms": r.total_latency_ms,
                     "cache_hit": bool(r.cache_hit), "is_repeat": wq.is_repeat, "probe_ms_per_token": ms})
    pd.DataFrame(rows).to_csv(OUT / "cag_dynamic_raw.csv", index=False)
    logger.info(f"done CAG dynamic: lat={pd.DataFrame(rows).latency_ms.mean():.0f}")


if __name__ == "__main__":
    for st in (sys.argv[1:] or ["smoke", "main", "concurrency", "dynamic", "paraphrase"]):
        logger.info(f"stage {st} started")
        {"smoke": stage_smoke, "main": stage_main, "concurrency": stage_concurrency,
         "dynamic": stage_dynamic, "paraphrase": stage_paraphrase}[st]()
        logger.info(f"stage {st} done")
    (OUT / "DONE.marker").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
