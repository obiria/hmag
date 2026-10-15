"""Driver of the component ablations, the capacity and locality sweeps, the sensitivity
analysis and the verifier evaluation.

  ablation1    every configuration of ablation.ABLATION_CONFIGS on the seed-1 stream of each dataset
  capacity     hit rate against total capacity and Zipf exponent
  sensitivity  one-factor-at-a-time parameter sweep
  verifier     SNLI evaluation of the NLI model and the numeric-contradiction test
  combine      summary files and strategy frequencies of the ablation runs

Every configuration runs on a freshly built HMAGSystem."""

import os

# eight intra-op threads for all configurations
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

OUT = Path(os.environ.get("REV_OUT", "./revision_results"))
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(OUT / "revision_run.log"), logging.StreamHandler()],
)
for noisy in ("httpx", "hmag_system", "hmag_core", "hmag_retrieval", "hmag_verifier",
              "sentence_transformers", "faiss", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("revision")

import torch  # noqa: E402

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

from hmag_system import HMAGConfig, HMAGSystem  # noqa: E402

PRIMARY_MODEL = "llama3.2:latest"
USE_GPU = os.environ.get("REV_USE_GPU", "0") == "1"
# GPU speed check: with REV_MAX_MS_PER_TOKEN set, a configuration starts only once a fixed
# 200-token generation decodes at no more than that many ms per token.
MAX_MS_PER_TOKEN = float(os.environ.get("REV_MAX_MS_PER_TOKEN", "0"))


def _wait_for_fast_gpu(model: str = "llama3.2:latest") -> float:
    if MAX_MS_PER_TOKEN <= 0:
        return float("nan")
    import requests
    while True:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": model, "prompt": "Write 200 words about the history of Normandy.", "stream": False,
            "options": {"num_predict": 200, "temperature": 0}}, timeout=300).json()
        ms = r["eval_duration"] / 1e6 / r["eval_count"]
        if ms <= MAX_MS_PER_TOKEN:
            return ms
        logger.warning(f"GPU slow: {ms:.1f} ms/token > {MAX_MS_PER_TOKEN}; waiting 60s")
        time.sleep(60)


# Time budget: no configuration starts after SOFT_DEADLINE, and a running stream stops at
# HARD_DEADLINE. The *_EPOCH variables are converted once to the monotonic clock, so that
# adjustments of the system clock cannot end a run early.
RUN_START = time.monotonic()
_EPOCH_OFFSET = RUN_START - time.time()
SOFT_DEADLINE = (float(os.environ["SOFT_DEADLINE_EPOCH"]) + _EPOCH_OFFSET
                 if "SOFT_DEADLINE_EPOCH" in os.environ else RUN_START + 4.6 * 3600)
HARD_DEADLINE = (float(os.environ["HARD_DEADLINE_EPOCH"]) + _EPOCH_OFFSET
                 if "HARD_DEADLINE_EPOCH" in os.environ else RUN_START + 4.85 * 3600)


def _out_of_time() -> bool:
    return time.monotonic() > SOFT_DEADLINE


DATASETS = ["squad_v2", "natural_questions", "ms_marco"]
NUM_QUERIES = int(os.environ.get("REV_NUM_QUERIES", "150"))


def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _shared_models(model: str):
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from hmag_retrieval import OllamaGenerator
    from hmag_verifier import FactExtractor
    return dict(
        generator=OllamaGenerator(model_name=model),
        verifier_encoder=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'),
        verifier_nli_model=CrossEncoder('cross-encoder/nli-deberta-v3-small'),
        verifier_fact_extractor=FactExtractor(),
    )


def _run_stream(system: HMAGSystem, workload, extra: Dict) -> pd.DataFrame:
    """Runs the query stream once and returns one row per query (warm-up queries are
    flagged and excluded in the analysis)."""
    from hmag_core import QueryContext, PolicyGradientController
    from hmag_qa_metrics import score as qa_score
    from hmag_profiling import ResourceMonitor

    probe = _wait_for_fast_gpu()
    monitor = ResourceMonitor()
    monitor.start()
    rows = []
    for i, wq in enumerate(workload.queries):
        if time.monotonic() > HARD_DEADLINE:
            logger.warning(f"hard deadline hit at query {i}/{len(workload.queries)}; truncating {extra}")
            break
        r = system.process_query(wq.query, context=QueryContext(query=wq.query, timestamp=time.time()))
        err = bool(r.metadata.get('generation_error'))
        bd = r.latency_breakdown or {}
        row = {
            **extra, 'query_index': i, 'is_warmup': wq.is_warmup, 'is_repeat': wq.is_repeat,
            'latency_ms': r.total_latency_ms, 'cache_hit': bool(r.cache_hit),
            'used_retrieval': any(d.source_type != "cache" for d in r.documents_used),
            'strategy': r.strategy_used.name if r.strategy_used is not None else None,
            'exact_match': bool(r.metadata.get('exact_match', False)),
            **qa_score(wq.reference_answer, r.response, err), 'response': r.response,
            'reference': wq.reference_answer,
            'consistency': r.consistency_result.confidence_score if r.consistency_result else float('nan'),
            'reward': r.metadata.get('reward', float('nan')),
            'prompt_tokens': r.metadata.get('prompt_tokens', float('nan')),
            'generated_tokens': r.metadata.get('generated_tokens', float('nan')),
            'error': err,
        }
        for k in ('cache', 'controller', 'retrieval', 'verification', 'time_to_first_token',
                  'decoding', 'generation', 'cache_update', 'exact_match'):
            row[f'lat_{k}_ms'] = bd.get(k, float('nan'))
        if isinstance(system.controller, PolicyGradientController):
            row['grad_norm'] = getattr(system.controller, 'last_grad_norm', float('nan'))
        rows.append(row)
    summary = monitor.stop()
    df = pd.DataFrame(rows)
    for f in ('avg_gpu_util_pct', 'peak_gpu_memory_mb', 'avg_gpu_power_w', 'energy_joules',
              'peak_rss_memory_mb', 'duration_s'):
        df[f] = getattr(summary, f)
    for t in range(3):
        df[f'tier{t + 1}_hit_rate'] = system.cache.stats.get_tier_hit_rate(t)
    df['probe_ms_per_token'] = probe
    return df


def _summarize(raw: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    from hmag_profiling import bootstrap_ci
    rows = []
    for k, g in raw.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        m = g[~g['is_warmup']]
        lat, lo, hi = bootstrap_ci(m['latency_ms'].tolist())
        q, qlo, qhi = bootstrap_ci(m['rouge_l'].tolist())
        per_run = g.groupby([c for c in ('dataset', 'seed') if c in g.columns]).first() \
            if any(c in g.columns for c in ('dataset', 'seed')) else g.iloc[[0]]
        rows.append({
            **dict(zip(keys, k)), 'n_measured': len(m), 'n_runs': len(per_run),
            'latency_ms': lat, 'latency_ci_lo': lo, 'latency_ci_hi': hi,
            'latency_p50_ms': float(m['latency_ms'].median()),
            'latency_p95_ms': float(m['latency_ms'].quantile(0.95)),
            'cache_hit_rate': float(m['cache_hit'].mean()),
            'retrieval_rate': float(m['used_retrieval'].mean()),
            'repeat_ceiling': float(m['is_repeat'].mean()),
            'rouge_l': q, 'rouge_l_ci_lo': qlo, 'rouge_l_ci_hi': qhi,
            'consistency': float(m['consistency'].mean()),
            'generated_tokens': float(m['generated_tokens'].mean()),
            'error_rate': float(m['error'].mean()),
            'peak_rss_gb': float(per_run['peak_rss_memory_mb'].max()) / 1024,
            'avg_gpu_util_pct': float(per_run['avg_gpu_util_pct'].mean()),
            'energy_j_per_query': float(per_run['energy_joules'].sum() / len(g)),
        })
    return pd.DataFrame(rows)


# Verifier stage
def _squad_contradictions(n: int, seed: int = 7):
    """(question, true sentence, contradicting sentence) triples built from
    SQuAD v2.0 validation: the sentence of the context containing a numeric
    gold answer, and the same sentence with that number replaced by a
    different value. The perturbed value is deterministic per item."""
    import re
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    rng = np.random.default_rng(seed)
    items, seen = [], set()
    for ex in ds:
        if not ex['answers']['text']:
            continue
        ans = ex['answers']['text'][0].strip()
        if not re.fullmatch(r"\d{1,4}(,\d{3})*", ans):
            continue
        sents = re.split(r"(?<=[.!?])\s+", ex['context'])
        sent = next((s for s in sents if re.search(rf"(?<![\d,]){re.escape(ans)}(?![\d,])", s)), None)
        if sent is None or sent in seen or len(sent) > 400:
            continue
        val = int(ans.replace(',', ''))
        if 1000 <= val <= 2100:  # years: shift by 7..40 years
            new = val + int(rng.choice([-1, 1])) * int(rng.integers(7, 41))
        else:
            new = max(val + 1, int(round(val * float(rng.choice([0.3, 0.5, 2.0, 3.0])))))
        new_s = f"{new:,}" if ',' in ans else str(new)
        if new_s == ans:
            continue
        false = re.sub(rf"(?<![\d,]){re.escape(ans)}(?![\d,])", new_s, sent, count=1)
        seen.add(sent)
        items.append({'query': ex['question'], 'true_doc': sent, 'false_doc': false, 'answer': ans})
        if len(items) >= n:
            break
    return items


def stage_verifier(model=PRIMARY_MODEL, nli_n=1000, contra_n=100):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import f1_score
    from hmag_core import QueryContext, RetrievalStrategy
    from hmag_retrieval import Document, HierarchicalRetriever, OllamaGenerator
    from hmag_verifier import (NLIConsistencyChecker, FactualConsistencyEvaluator,
                               AdvancedVerifier, FactExtractor)
    from hmag_verifier_eval import _load_nli_eval_set, NLI_LABELS
    from hmag_profiling import bootstrap_ci

    nli = NLIConsistencyChecker()
    fc = FactualConsistencyEvaluator(nli_checker=nli)
    enc = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    fx = FactExtractor()

    eval_set = _load_nli_eval_set(nli_n)
    probs = nli.predict([p for p, _, _ in eval_set], [h for _, h, _ in eval_set])
    pred = [NLI_LABELS[i] for i in np.argmax(probs, axis=1)]
    gold = [lab for _, _, lab in eval_set]
    macro_f1 = float(f1_score(gold, pred, labels=NLI_LABELS, average="macro"))
    rng = np.random.default_rng(0)
    boots = []
    g_arr, p_arr = np.array(gold), np.array(pred)
    for _ in range(2000):
        idx = rng.integers(0, len(gold), len(gold))
        boots.append(f1_score(g_arr[idx], p_arr[idx], labels=NLI_LABELS, average="macro"))
    f1_lo, f1_hi = np.percentile(boots, [2.5, 97.5])
    per_class = f1_score(gold, pred, labels=NLI_LABELS, average=None)
    logger.info(f"verifier/SNLI n={len(gold)} macroF1={macro_f1:.3f} [{f1_lo:.3f},{f1_hi:.3f}] "
                f"per-class={dict(zip(NLI_LABELS, np.round(per_class, 3)))}")

    items = _squad_contradictions(contra_n)
    logger.info(f"verifier: {len(items)} SQuAD-derived contradiction items")
    retriever = HierarchicalRetriever(use_gpu=USE_GPU)
    retriever.add_documents([Document(doc_id=f"true_{i}", content=it['true_doc']) for i, it in enumerate(items)])
    generator = OllamaGenerator(model_name=model)

    hybrid = list(RetrievalStrategy).index(RetrievalStrategy.HYBRID)
    base = {"controller_type": "fixed", "fixed_strategy_idx": hybrid}
    configs = {
        "Without verifier": {**base, "enable_verification": False, "verifier_mode": "full"},
        "NLI-based verifier": {**base, "enable_verification": True, "verifier_mode": "nli_only"},
        "Full HMAG verifier": {**base, "enable_verification": True, "verifier_mode": "full"},
    }
    rows, item_rows = [], []
    for name, ov in configs.items():
        cfg = HMAGConfig(ollama_model=model)
        for k, v in ov.items():
            setattr(cfg, k, v)
        system = HMAGSystem(cfg, retriever=retriever, generator=generator, verifier_encoder=enc,
                            verifier_nli_model=nli.model, verifier_fact_extractor=fx)
        fcs, contains = [], []
        for it in items:
            r = system.process_query(it['query'], context=QueryContext(query=it['query'], timestamp=time.time()))
            ctx = [d.content for d in r.documents_used if d.metadata.get('entry_type') != 'qa_response']
            if not ctx:
                score = float('nan')  # no documents in the prompt
            elif not (r.response or '').strip():
                score = 0.0
            else:
                score = fc.score(r.response, ctx).entailed_fraction
            fcs.append(score)
            contains.append(it['answer'] in (r.response or ''))
            item_rows.append({'config': name, 'query': it['query'], 'factual_consistency': score,
                              'answer_in_response': contains[-1], 'latency_ms': r.total_latency_ms})
        detected, lats, false_alarms = 0, [], 0
        if name != "Without verifier":
            ver = AdvancedVerifier(consistency_threshold=cfg.consistency_threshold,
                                   contradiction_threshold=cfg.contradiction_threshold,
                                   mode=cfg.verifier_mode, encoder=enc, nli_model=nli.model,
                                   fact_extractor=fx)
            for it in items:
                vr = ver.verify_consistency([it['false_doc']], [it['true_doc']], it['query'])
                lats.append(vr.latency_ms)
                detected += int((not vr.is_consistent) or bool(vr.contradictions))
                # false-alarm control: an identical pair must not be flagged
                vc = ver.verify_consistency([it['true_doc']], [it['true_doc']], it['query'])
                false_alarms += int((not vc.is_consistent) or bool(vc.contradictions))
        fc_m, fc_lo, fc_hi = bootstrap_ci(fcs)
        rows.append({
            'Configuration': name,
            'SNLI macro-F1': macro_f1 if name != "Without verifier" else float('nan'),
            'SNLI F1 CI': f"[{f1_lo:.3f},{f1_hi:.3f}]" if name != "Without verifier" else '',
            'Factual Consistency': fc_m, 'FC CI lo': fc_lo, 'FC CI hi': fc_hi,
            'Answer EM-in-response': float(np.mean(contains)),
            'Conflict Detection Rate': detected / len(items) if name != "Without verifier" else float('nan'),
            'False Alarm Rate': false_alarms / len(items) if name != "Without verifier" else float('nan'),
            'Verification Latency (ms)': float(np.mean(lats)) if lats else float('nan'),
            'n_snli': len(gold), 'n_items': len(items),
        })
        logger.info(f"verifier/{name}: {rows[-1]}")
    ts = _ts()
    pd.DataFrame(rows).to_csv(OUT / f"table_viii_{ts}.csv", index=False)
    pd.DataFrame(item_rows).to_csv(OUT / f"table_viii_items_{ts}.csv", index=False)


# Ablation stage
def stage_ablation(seeds=(1,), model=PRIMARY_MODEL, num_queries=NUM_QUERIES, datasets=None, only=None):
    from ablation import ABLATION_CONFIGS
    from hmag_benchmark import get_workload
    from hmag_retrieval import HierarchicalRetriever

    shared = _shared_models(model)
    for seed in seeds:
        for ds in (datasets or DATASETS):
            wl = get_workload(ds, num_queries, seed)
            retriever = HierarchicalRetriever(use_gpu=USE_GPU)
            if wl.corpus:
                retriever.add_documents(wl.corpus)
            parts = []
            for name, ov in ABLATION_CONFIGS.items():
                if only is not None and name not in only:
                    continue
                if _out_of_time():
                    logger.warning(f"soft deadline: skipping ablation {ds} s{seed} {name}")
                    continue
                t0 = time.time()
                retriever._query_embed_cache.clear()
                try:
                    cfg = HMAGConfig(ollama_model=model)
                    for k, v in ov.items():
                        setattr(cfg, k, v)
                    system = HMAGSystem(cfg, retriever=retriever, **shared)
                    df = _run_stream(system, wl, {'ablation': name, 'dataset': wl.name,
                                                  'seed': seed, 'model': model})
                    parts.append(df)
                    m = df[~df['is_warmup']]
                    logger.info(f"ablation {ds} s{seed} {name}: lat={m['latency_ms'].mean():.0f}ms "
                                f"hit={m['cache_hit'].mean():.3f} rouge={m['rouge_l'].mean():.3f} "
                                f"({time.time() - t0:.0f}s)")
                except Exception as e:
                    logger.error(f"ablation {ds} s{seed} {name} crashed: {e}\n{traceback.format_exc()}")
            if parts:
                pd.concat(parts, ignore_index=True).to_csv(OUT / f"ablation_raw_{ds}_s{seed}.csv", index=False)
    combine_ablation()


def combine_ablation():
    files = sorted(OUT.glob("ablation_raw_*_s*.csv"))
    if not files:
        return
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ts = _ts()
    _summarize(raw, ['ablation']).to_csv(OUT / f"ablation_summary_overall_{ts}.csv", index=False)
    _summarize(raw, ['ablation', 'dataset']).to_csv(OUT / f"ablation_summary_by_dataset_{ts}.csv", index=False)
    full = raw[(raw['ablation'] == 'full_system') & (~raw['is_warmup'])]
    strat = full.assign(bucket=(full['query_index'] // 25) * 25).groupby(['bucket', 'strategy']).size() \
        .unstack(fill_value=0)
    strat.div(strat.sum(axis=1), axis=0).to_csv(OUT / f"strategy_distribution_{ts}.csv")
    logger.info(f"combined ablation from {len(files)} files")


# Sensitivity stage
def stage_sensitivity(model=PRIMARY_MODEL, num_queries=int(os.environ.get("REV_SENS_QUERIES", "50")), seed=1):
    from hmag_sensitivity import DEFAULT_SWEEP_GRID
    from hmag_benchmark import get_workload
    from hmag_retrieval import HierarchicalRetriever

    shared = _shared_models(model)
    wl = get_workload("squad_v2", num_queries, seed)
    retriever = HierarchicalRetriever(use_gpu=USE_GPU)
    retriever.add_documents(wl.corpus)
    parts = []
    for field, values in DEFAULT_SWEEP_GRID.items():
        for v in values:
            if _out_of_time():
                logger.warning(f"soft deadline: skipping sensitivity {field}={v}")
                continue
            retriever._query_embed_cache.clear()
            cfg = HMAGConfig(ollama_model=model)
            setattr(cfg, field, v)
            system = HMAGSystem(cfg, retriever=retriever, **shared)
            df = _run_stream(system, wl, {'parameter': field, 'value': v, 'dataset': wl.name, 'seed': seed})
            parts.append(df)
            m = df[~df['is_warmup']]
            logger.info(f"sensitivity {field}={v}: lat={m['latency_ms'].mean():.0f} hit={m['cache_hit'].mean():.3f} "
                        f"rouge={m['rouge_l'].mean():.3f}")
            pd.concat(parts, ignore_index=True).to_csv(OUT / "sensitivity_raw.csv", index=False)
    if not parts:
        return
    raw = pd.concat(parts, ignore_index=True)
    _summarize(raw, ['parameter', 'value']).to_csv(OUT / f"sensitivity_summary_{_ts()}.csv", index=False)


# Capacity stage
def _oracle_bound(C: int, N: int, alpha: float) -> float:
    i = np.arange(1, N + 1, dtype=float)
    w = i ** (-alpha)
    return float(w[:min(C, N)].sum() / w.sum())


def stage_capacity(model=PRIMARY_MODEL, num_queries=int(os.environ.get("REV_CAP_QUERIES", "100")), seed=1):
    from hmag_datasets import load_squad_v2
    from hmag_retrieval import HierarchicalRetriever

    shared = _shared_models(model)
    parts, rows = [], []

    def run(tag, alpha, caps):
        wl = load_squad_v2(num_queries=num_queries, alpha=alpha, seed=seed)
        retriever = HierarchicalRetriever(use_gpu=USE_GPU)
        retriever.add_documents(wl.corpus)
        N = wl.metadata['raw_query_pool']
        for C in caps:
            if _out_of_time():
                logger.warning(f"soft deadline: skipping {tag} alpha={alpha} C={C}")
                continue
            retriever._query_embed_cache.clear()
            cfg = HMAGConfig(ollama_model=model)
            if C is not None:
                cfg.l1_cache_size = max(1, round(C * 0.1))
                cfg.l2_cache_size = max(1, round(C * 0.3))
                cfg.l3_cache_size = max(1, C - cfg.l1_cache_size - cfg.l2_cache_size)
            Ctot = cfg.l1_cache_size + cfg.l2_cache_size + cfg.l3_cache_size
            system = HMAGSystem(cfg, retriever=retriever, **shared)
            df = _run_stream(system, wl, {'sweep': tag, 'alpha': alpha, 'capacity': Ctot, 'seed': seed})
            parts.append(df)
            m = df[~df['is_warmup']]
            rows.append({'sweep': tag, 'alpha': alpha, 'capacity': Ctot, 'pool_N': N,
                         'distinct_in_stream': len({q.query for q in wl.queries}),
                         'oracle_bound': _oracle_bound(Ctot, N, alpha),
                         'repeat_ceiling': float(m['is_repeat'].mean()),
                         'hit_rate': float(m['cache_hit'].mean()),
                         'latency_ms': float(m['latency_ms'].mean()),
                         'rouge_l': float(m['rouge_l'].mean())})
            logger.info(f"capacity {rows[-1]}")
            pd.DataFrame(rows).to_csv(OUT / "capacity_alpha_summary.csv", index=False)

    # REV_CAP_SIZES restricts the capacity sweep (e.g. "5,10,25,50,100") and skips the
    # locality sweep; rows of configurations that are not rerun are kept from the latest run.
    only = [int(c) for c in os.environ.get("REV_CAP_SIZES", "").split(",") if c]
    previous = sorted(OUT.glob("capacity_alpha_raw_*.csv"))
    old_summary = pd.read_csv(OUT / "capacity_alpha_summary.csv") if \
        (only and (OUT / "capacity_alpha_summary.csv").exists()) else None
    if only:
        run('capacity', 1.2, only)
    else:
        run('capacity', 1.2, [5, 10, 25, 50, 100, None])
        for a in [0.6, 0.9, 1.5, 1.8]:
            run('alpha', a, [None])
    if not parts:
        return
    new = pd.concat(parts, ignore_index=True)
    summary = pd.DataFrame(rows)
    if only and previous:
        old = pd.read_csv(previous[-1])
        rerun = set(zip(new['sweep'], new['capacity']))
        old = old[[(s, c) not in rerun for s, c in zip(old['sweep'], old['capacity'])]]
        new = pd.concat([old, new], ignore_index=True)
        if old_summary is not None:
            keep = [(s, c) not in rerun for s, c in zip(old_summary['sweep'], old_summary['capacity'])]
            summary = pd.concat([old_summary[keep], summary], ignore_index=True)
    new.to_csv(OUT / f"capacity_alpha_raw_{_ts()}.csv", index=False)
    summary.to_csv(OUT / "capacity_alpha_summary.csv", index=False)


STAGES = {
    'verifier': stage_verifier,
    'ablation1': lambda: stage_ablation(seeds=(1,)),
    'capacity': stage_capacity,
    'sensitivity': stage_sensitivity,
    'combine': combine_ablation,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", choices=list(STAGES), default=['ablation1', 'capacity', 'sensitivity', 'verifier', 'combine'])
    args = ap.parse_args()
    for s in args.stages:
        logger.info(f"stage {s} started")
        t0 = time.time()
        try:
            STAGES[s]()
            logger.info(f"stage {s} done ({(time.time() - t0) / 60:.1f} min)")
        except Exception as e:
            logger.error(f"stage {s} failed: {e}\n{traceback.format_exc()}")
    (OUT / "REVISION_RUN_DONE.marker").write_text(time.strftime('%Y-%m-%d %H:%M:%S\n'))
