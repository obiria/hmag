"""
Paraphrase workloads: 40% verbatim repeats, 35% paraphrases of earlier questions and
25% new questions, built from SQuAD v2.0, Natural Questions and MS MARCO questions and
their reference answers.

Paraphrase bank: for each dataset, up to N_BASES distinct questions (with
reference answers) are paraphrased by a different LLM from the generator
under test (llama3:latest, 8B; the generator is llama3.2:latest). A
paraphrase is kept only if it differs from the original after
normalisation and the DeBERTa-v3 NLI cross-encoder judges the two to
entail each other in both directions (p_entail >= 0.5 each way). The bank
is written to JSON once and reused, so every method and seed sees the same
paraphrases.

Stream construction (per seed, 500 queries): the first query is a new
question; afterwards each position is a new question with probability
0.25, a verbatim repeat of an already-issued question (uniform over issued
questions) with probability 0.40, or an unused paraphrase of an
already-issued question with probability 0.35 (a new question if no unused
paraphrase remains). A paraphrase inherits its source question's reference
answer and gold document. The first 10% of positions are flagged warm-up.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests

from hmag_datasets import WorkloadQuery, RAGWorkload, load_squad_v2, load_natural_questions, load_ms_marco

logger = logging.getLogger(__name__)

BANK_DIR = Path("./revision_results/paraphrase_bank")
N_BASES = 300
K_PARAPHRASES = 3
PARAPHRASER = "llama3:latest"
LOADERS = {"squad_v2": load_squad_v2, "natural_questions": load_natural_questions, "ms_marco": load_ms_marco}
P_NEW, P_REPEAT = 0.25, 0.40


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _base_pool(dataset: str):
    """Distinct questions with reference answers, plus the corpus, from the
    dataset's standard loader (uniform order over its raw pool)."""
    wl = LOADERS[dataset](num_queries=4 * N_BASES, alpha=0.0, seed=0)
    seen, bases = set(), []
    for q in wl.queries:
        if q.reference_answer and q.query not in seen:
            seen.add(q.query)
            bases.append({"query": q.query, "answer": q.reference_answer, "gold": q.gold_doc_ids})
        if len(bases) >= N_BASES:
            break
    return bases, wl.corpus


def _paraphrase(question: str) -> List[str]:
    prompt = (f"Rewrite the following question in {K_PARAPHRASES} different ways. Keep exactly the same meaning "
              f"and the same answer; change the wording and sentence structure. Output only the {K_PARAPHRASES} "
              f"rewritten questions, one per line, with no numbering or extra text.\n\nQuestion: {question}")
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": PARAPHRASER, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.7, "num_predict": 200}}, timeout=300).json()
    lines = [re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", l).strip() for l in r.get("response", "").splitlines()]
    return [l for l in lines if l.endswith("?") and len(l) > 10][:K_PARAPHRASES]


def build_bank(dataset: str) -> Path:
    from sentence_transformers import CrossEncoder
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    out = BANK_DIR / f"{dataset}.json"
    if out.exists():
        return out
    nli = CrossEncoder("cross-encoder/nli-deberta-v3-small")  # labels: contradiction, entailment, neutral
    bases, _ = _base_pool(dataset)
    kept_total, cand_total = 0, 0
    for b in bases:
        cands = [c for c in _paraphrase(b["query"]) if _norm(c) != _norm(b["query"])]
        cand_total += len(cands)
        if cands:
            fwd = nli.predict([(b["query"], c) for c in cands], apply_softmax=True)[:, 1]
            bwd = nli.predict([(c, b["query"]) for c in cands], apply_softmax=True)[:, 1]
            b["paraphrases"] = [c for c, f, g in zip(cands, fwd, bwd) if f >= 0.5 and g >= 0.5]
        else:
            b["paraphrases"] = []
        kept_total += len(b["paraphrases"])
    json.dump({"dataset": dataset, "paraphraser": PARAPHRASER, "candidates": cand_total,
               "kept": kept_total, "bases": bases}, open(out, "w"), indent=1)
    logger.info(f"{dataset}: kept {kept_total}/{cand_total} paraphrases for {len(bases)} questions")
    return out


def paraphrase_workload(dataset: str, num_queries: int = 500, seed: int = 1,
                        warmup_fraction: float = 0.1) -> RAGWorkload:
    bank = json.load(open(BANK_DIR / f"{dataset}.json"))
    bases = [b for b in bank["bases"]]
    _, corpus = _base_pool(dataset)
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(bases)))
    issued: List[int] = []
    unused: Dict[int, List[str]] = {}
    seen_text = set()
    queries, kinds = [], []
    for t in range(num_queries):
        u = rng.random()
        kind = "new" if (t == 0 or u < P_NEW) else ("repeat" if u < P_NEW + P_REPEAT else "paraphrase")
        if kind == "paraphrase":
            cands = [i for i in issued if unused.get(i)]
            if not cands:
                kind = "new"
        if kind == "new" and not order:
            kind = "repeat"
        if kind == "new":
            i = order.pop(0)
            issued.append(i)
            unused[i] = list(bases[i]["paraphrases"])
            text = bases[i]["query"]
        elif kind == "repeat":
            i = issued[int(rng.integers(len(issued)))]
            text = bases[i]["query"]
        else:
            i = cands[int(rng.integers(len(cands)))]
            text = unused[i].pop(int(rng.integers(len(unused[i]))))
        queries.append(WorkloadQuery(query=text, reference_answer=bases[i]["answer"], gold_doc_ids=bases[i]["gold"],
                                     is_warmup=t < int(num_queries * warmup_fraction), is_repeat=text in seen_text))
        kinds.append(kind)
        seen_text.add(text)
    return RAGWorkload(name=f"{dataset}_paraphrase", corpus=corpus, queries=queries,
                       metadata={"kinds": kinds, "seed": seed, "paraphraser": PARAPHRASER,
                                 "mix": {"new": P_NEW, "repeat": P_REPEAT, "paraphrase": 1 - P_NEW - P_REPEAT}})


P_REWORD_ZIPF = 0.35


def zipf_paraphrase_workload(dataset: str, num_queries: int = 500, seed: int = 1, alpha: float = 1.2,
                             p_reword: float = P_REWORD_ZIPF, warmup_fraction: float = 0.1) -> RAGWorkload:
    """Stream with skewed popularity and reworded repeats: questions of the paraphrase bank are
    drawn with Zipf(alpha) popularity (bank order shuffled with the seed). The first request
    for a question uses its original wording; every later request is one of its paraphrases
    with probability `p_reword` and the original wording otherwise. Reference answers are
    those of the source question."""
    from hmag_datasets import _power_law_order
    bank = json.load(open(BANK_DIR / f"{dataset}.json"))
    bases = [b for b in bank["bases"] if b["paraphrases"]]
    _, corpus = _base_pool(dataset)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(bases))
    draws = _power_law_order(len(bases), num_queries, alpha, seed)
    asked, seen_text = set(), set()
    queries, kinds = [], []
    for t, r in enumerate(draws):
        b = bases[int(order[int(r)])]
        i = int(order[int(r)])
        if i not in asked:
            kind, text = "new", b["query"]
            asked.add(i)
        elif rng.random() < p_reword:
            kind, text = "paraphrase", b["paraphrases"][int(rng.integers(len(b["paraphrases"])))]
        else:
            kind, text = "repeat", b["query"]
        queries.append(WorkloadQuery(query=text, reference_answer=b["answer"], gold_doc_ids=b["gold"],
                                     is_warmup=t < int(num_queries * warmup_fraction), is_repeat=text in seen_text))
        kinds.append(kind)
        seen_text.add(text)
    return RAGWorkload(name=f"{dataset}_zipf_paraphrase", corpus=corpus, queries=queries,
                       metadata={"kinds": kinds, "seed": seed, "alpha": alpha, "p_reword": p_reword,
                                 "paraphraser": PARAPHRASER, "bank_questions": len(bases)})
