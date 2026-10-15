"""Tier placement and search cost of HMAG's answer cache in main memory, cache only (no retrieval
or generation), against HMAG (original placement), which stores new answers in L1, and a flat cache of the
same capacity with the same priority score, admission test and exact index.

Configurations (all with admission, exact slot indexes, HierarchicalCache):
  HMAG             tiers 1,500 : 2,100 : 7,500, placement "hot", log frequency, promotion
                   threshold 0.5, direct promotion to L1, no sweep (the defaults of HMAGConfig)
  HMAG (original)  tiers 1 : 10 : 100, placement "insert_l1", exp frequency, promotion
                   threshold 0.7, periodic sweep
  Flat             one tier of the same total capacity, HMAG's priority and admission

Pool: the 101,093 distinct MS MARCO v2.1 validation questions (embeddings in
revision_results/progressive_data/emb_msmarco_pool.npy). A stream of STREAM queries draws from
them with Zipf popularity (alpha = 1.2 as in the main streams, and 0.8). The questions of the
paraphrase banks take the place of pool questions drawn from the stream's requests, so they
have the stream's popularity mix, and each of their requests is a paraphrase with probability
0.5, so that rewordings arrive as often as their question is requested.

Request sequence of HMAGSystem.process_query: exact lookup, semantic search on a miss, storage
of the answer; the tier promotions of a hit are applied after the lookup (maintain). Lookup
time (exact lookup plus search) and management time (promotion, admission, insertion,
eviction and index updates) are measured separately with perf_counter. Search quality: for
every semantic hit, the best match in any tier (top-1 of each tier's index) is compared with
the returned match.

Usage: OMP_NUM_THREADS=1 python placement_benchmark.py
Output: revision_results/placement/summary.csv and raw/<run>.npz."""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hmag_core import HierarchicalCache
from hmag_datasets import _local_cached_file, _power_law_order
from hmag_system import HMAGConfig

OUT = Path(os.environ.get("PLACE_OUT", "revision_results/placement"))
DATA = Path("revision_results/progressive_data")
STREAM = int(os.environ.get("PLACE_STREAM", "40000"))
ALPHAS = [float(a) for a in os.environ.get("PLACE_ALPHAS", "1.2,0.8").split(",")]
SIZES = [int(c) for c in os.environ.get("PLACE_SIZES", "11100,111000").split(",")]
WARMUP, STEP_S, DIM, ITEM, P_REWORD = 0.2, 0.33, 768, 256, 0.5
BETTER = 0.02  # a match this much more similar in a lower tier counts as a weaker early match


class VirtualClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def configs(C):
    """HMAG with the default split (1,500 / 2,100 / 7,500 at C = 11,100, scaled with C), the tier
    management and split of HMAG (original placement) (1 : 10 : 100), and a flat cache."""
    l1, l2 = round(C * 1500 / 11100), round(C * 2100 / 11100)
    u = C // 111
    hmag = dict(placement="hot", frequency_norm="log", promote_threshold=0.5, promote_to_top=True,
                promote_sweep_interval=10 ** 9)
    return {"HMAG": ((l1, l2, C - l1 - l2), hmag),
            "HMAG (original placement)": ((u, 10 * u, C - 11 * u),
                                          dict(placement="insert_l1", frequency_norm="exp", promote_threshold=0.7)),
            "Flat": ((C, 0, 0), dict(hmag, promote_to_top=False))}


def pool_questions(seed=0):
    import pyarrow.parquet as pq
    path = _local_cached_file("microsoft/ms_marco", "v2.1/validation-00000-of-00001.parquet")
    questions = list(dict.fromkeys(pq.read_table(path, columns=["query"]).column("query").to_pylist()))
    order = np.random.default_rng(seed).permutation(len(questions))
    return [questions[i] for i in order]


def encode(texts, path):
    if path.exists():
        e = np.load(path)
        if len(e) == len(texts):
            return e
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(HMAGConfig().retrieval_model, device="cpu")
    e = enc.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False)
    e = (e / np.linalg.norm(e, axis=1, keepdims=True)).astype("float32")
    np.save(path, e)
    return e


def bank():
    bases = []
    for d in ("squad_v2", "natural_questions", "ms_marco"):
        bases += json.load(open(Path("revision_results/paraphrase_bank") / f"{d}.json"))["bases"]
    return [b for b in bases if b["paraphrases"]]


def mixed_stream(questions, base_stream, bases, seed=1):
    rng = np.random.default_rng(seed)
    replaced = {}
    for b in bases:
        while True:
            qi = int(base_stream[int(rng.integers(len(base_stream)))])
            if qi not in replaced:
                replaced[qi] = b
                break
    texts, kinds = [], []
    for qi in base_stream:
        b = replaced.get(int(qi))
        if b is None:
            texts.append(questions[qi]); kinds.append("background")
        elif rng.random() < P_REWORD:
            texts.append(b["paraphrases"][int(rng.integers(len(b["paraphrases"])))]); kinds.append("paraphrase")
        else:
            texts.append(b["query"]); kinds.append("bank_source")
    return texts, np.array(kinds)


def run_one(name, tiers, opts, stream, kinds, emb, index_of, tag):
    cfg = HMAGConfig()
    clock = VirtualClock()
    np.random.seed(1)
    cache = HierarchicalCache(l1_size=tiers[0], l2_size=tiers[1], l3_size=tiers[2], embedding_dim=DIM,
                              similarity_thresholds=(cfg.similarity_threshold,) * 3, n_min=cfg.cache_n_min,
                              priority_weights=cfg.priority_weights, demote_threshold=cfg.demote_threshold,
                              diversity_lambda=cfg.diversity_lambda, encoder=object(), clock=clock,
                              admission_filter=True, **opts)
    cache.encode_cached = lambda text: emb[index_of[text]]
    content = "a" * ITEM
    n = len(stream)
    hit, semantic = np.zeros(n, bool), np.zeros(n, bool)
    tier = np.full(n, -1, np.int8)
    lookup_us, manage_us = np.zeros(n, np.float32), np.zeros(n, np.float32)
    weaker = np.zeros(n, bool)
    perf = time.perf_counter
    t_start = perf()
    for i, q in enumerate(stream):
        key = hashlib.sha256(q.encode()).hexdigest()
        where = cache.tier_of(key)
        t0 = perf()
        entry = cache.get(key)
        if entry is not None:
            dt = perf() - t0
            hit[i], tier[i] = True, where
        else:
            matches, h, t = cache.search(q, k=cfg.retrieval_top_k, entry_type_filter="qa_response")
            dt = perf() - t0
            hit[i], semantic[i], tier[i] = h, h, (t if h else -1)
            if h:  # not timed: best match of every tier against the returned one
                qe = emb[index_of[q]]
                best = max((s for j in range(3) for _, s in cache.indexes[j].search(qe, 1)), default=-1.0)
                weaker[i] = best - float(qe @ matches[0].embedding) >= BETTER
        lookup_us[i] = dt * 1e6
        clock.now += STEP_S
        t1 = perf()
        cache.maintain()
        if entry is None:
            cache.put(key=key, content=content, embedding_text=q, entry_type="qa_response")
        manage_us[i] = (perf() - t1) * 1e6
    wall = perf() - t_start
    m = np.arange(n) >= int(WARMUP * n)
    h, s, lat, mg = hit[m], semantic[m], lookup_us[m], manage_us[m]
    v = h & ~s  # verbatim hits
    pct = lambda x, p: float(np.percentile(x, p)) if len(x) else float("nan")
    row = dict(config=name, l1=tiers[0], l2=tiers[1], l3=tiers[2], capacity=sum(tiers), queries=int(m.sum()),
               hit_rate=h.mean(), verbatim_hit_rate=v.mean(), semantic_hit_rate=s.mean(),
               hits_from_L1=float((tier[m][h] == 0).mean()), hits_from_L12=float((tier[m][h] <= 1).mean()),
               semantic_hits=int(s.sum()), semantic_from_L1=float((tier[m][s] == 0).mean()) if s.any() else np.nan,
               semantic_from_L12=float((tier[m][s] <= 1).mean()) if s.any() else np.nan,
               verbatim_hit_us_p50=pct(lat[v], 50),
               semantic_hit_us_mean=float(lat[s].mean()) if s.any() else np.nan, semantic_hit_us_p50=pct(lat[s], 50),
               semantic_hit_us_p99=pct(lat[s], 99), miss_us_mean=float(lat[~h].mean()), miss_us_p50=pct(lat[~h], 50),
               lookup_us_mean=float(lat.mean()), management_us_mean=float(mg.mean()),
               cache_us_mean=float((lat + mg).mean()),
               weaker_match_share=float(weaker[m][s].mean()) if s.any() else np.nan,
               paraphrase_hit_rate=float(h[kinds[m] == "paraphrase"].mean()),
               rejected=cache.rejected_admissions, run_s=wall)
    np.savez_compressed(OUT / "raw" / f"{tag}.npz", hit=hit, semantic=semantic, tier=tier,
                        lookup_us=lookup_us, manage_us=manage_us)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(exist_ok=True)
    questions = pool_questions()
    emb = encode(questions, DATA / "emb_msmarco_pool.npy")
    bases = bank()
    btexts = list(dict.fromkeys([b["query"] for b in bases] + [p for b in bases for p in b["paraphrases"]]))
    emb = np.concatenate([emb, encode(btexts, DATA / "emb_bank.npy")])
    index_of = {q: i for i, q in enumerate(questions)}
    for j, t in enumerate(btexts):
        index_of.setdefault(t, len(questions) + j)
    summary = OUT / "summary.csv"
    done = set(zip(pd.read_csv(summary).alpha, pd.read_csv(summary).capacity, pd.read_csv(summary).config)) \
        if summary.exists() else set()
    for alpha in ALPHAS:
        stream, kinds = mixed_stream(questions, _power_law_order(len(questions), STREAM, alpha, 1), bases)
        seen, rep = set(), np.zeros(len(stream), bool)
        for i, q in enumerate(stream):
            rep[i] = q in seen
            seen.add(q)
        for C in SIZES:
            for name, (tiers, opts) in configs(C).items():
                if (alpha, C, name) in done:
                    continue
                print(f"{time.strftime('%H:%M:%S')} alpha={alpha} C={C} {name} tiers={tiers}", flush=True)
                row = run_one(name, tiers, opts, stream, kinds, emb, index_of,
                              f"a{alpha}_C{C}_{name.split()[0]}_{'orig' if 'original' in name else 'x'}")
                row.update(alpha=alpha, stream=STREAM, repeat_share=float(rep[int(WARMUP * len(stream)):].mean()),
                           **{k: str(v) for k, v in opts.items()})
                print("  " + json.dumps({k: round(float(v), 4) for k, v in row.items()
                                         if isinstance(v, (int, float, np.floating, np.integer))}), flush=True)
                prev = [pd.read_csv(summary)] if summary.exists() else []
                pd.concat(prev + [pd.DataFrame([row])], ignore_index=True).to_csv(summary, index=False)


if __name__ == "__main__":
    main()
