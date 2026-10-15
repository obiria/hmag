"""
Generation prompt and answer-quality metrics shared by HMAG and every baseline.

The system prompt asks for a short answer, which is the standard setting for
extractive question-answering benchmarks (SQuAD v2.0, Natural Questions,
MS MARCO), whose reference answers are short spans or sentences.
Metrics follow the SQuAD evaluation script: answers are lower-cased and
punctuation, articles and extra whitespace are removed before exact match and
token-level F1 are computed. `contains` reports whether the normalized
reference occurs in the normalized response. ROUGE-L F1 is computed with the
rouge-score package (Google Research, version 0.1.2), whose tokenizer
lower-cases the text and removes punctuation.
"""
import re
import string
from collections import Counter

from rouge_score import rouge_scorer

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

SYSTEM_PROMPT = ("You are a question-answering assistant. Use the provided context to answer the question. "
                 "Reply with only the answer as a short phrase, without explanation. If the context does not "
                 "contain the answer, give your best short answer.")


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(reference: str, response: str) -> float:
    return float(normalize(reference) == normalize(response))


def token_f1(reference: str, response: str) -> float:
    ref, hyp = normalize(reference).split(), normalize(response).split()
    if not ref or not hyp:
        return float(ref == hyp)
    common = Counter(ref) & Counter(hyp)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    p, r = overlap / len(hyp), overlap / len(ref)
    return 2 * p * r / (p + r)


def contains(reference: str, response: str) -> float:
    ref = normalize(reference)
    return float(bool(ref) and f" {ref} " in f" {normalize(response)} ")


def score(reference, response, error=False) -> dict:
    """All quality metrics for one answer. Questions without a reference are not scored (NaN); a failed
    generation or an empty answer to a question with a reference scores zero, so that failures stay in the
    denominator instead of being dropped."""
    if not reference:
        nan = float("nan")
        return {"rouge_l": nan, "exact_match_score": nan, "token_f1": nan, "gold_contained": nan}
    if error or not response or not str(response).strip():
        return {"rouge_l": 0.0, "exact_match_score": 0.0, "token_f1": 0.0, "gold_contained": 0.0}
    return {"rouge_l": _ROUGE.score(reference, response)["rougeL"].fmeasure, "exact_match_score": exact_match(reference, response),
            "token_f1": token_f1(reference, response), "gold_contained": contains(reference, response)}
