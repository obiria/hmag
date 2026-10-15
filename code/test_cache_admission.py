"""Smoke tests of the cache admission filter (Algorithm 1). Synthetic embeddings, no GPU
or model server needed. Run: python -m pytest -q test_cache_admission.py"""

import hashlib

import numpy as np

from hmag_core import HierarchicalCache, RequestFrequency

DIM = 768


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make_cache(tiers, admission, clock=None):
    cache = HierarchicalCache(l1_size=tiers[0], l2_size=tiers[1], l3_size=tiers[2], embedding_dim=DIM,
                              encoder=object(), admission_filter=admission, clock=clock or Clock())
    rng = np.random.default_rng(0)
    vectors = {}

    def encode(text):  # distinct questions get near-orthogonal random unit vectors
        if text not in vectors:
            v = rng.standard_normal(DIM).astype("float32")
            vectors[text] = v / np.linalg.norm(v)
        return vectors[text]

    cache.encode_cached = encode
    return cache


def request(cache, question, clock=None):
    """One query as HMAGSystem processes it; returns True on a hit."""
    key = hashlib.sha256(question.encode()).hexdigest()
    entry = cache.get(key)
    if entry is not None:
        hit = True
    else:
        _, hit, _ = cache.search(question, k=10, entry_type_filter="qa_response")
        cache.put(key=key, content="answer", embedding_text=question, entry_type="qa_response")
    if clock is not None:
        clock.now += 0.3
    return hit


def zipf_stream(n, pool, alpha, seed):
    rng = np.random.default_rng(seed)
    p = np.arange(1, pool + 1, dtype=float) ** -alpha
    return [f"q{i}" for i in rng.choice(pool, size=n, p=p / p.sum())]


def placement(cache):
    return [sorted(t.keys()) for t in cache.tiers]


def test_request_frequency_ages_and_prunes():
    f = RequestFrequency(sample_size=4)
    for key in ["a", "a", "a", "b"]:
        f.record(key)
    assert f.estimate("a") == 1 and f.estimate("b") == 0  # halved after 4 records
    assert "b" not in f.counts


def test_filter_has_no_effect_while_cache_has_room():
    stream = zipf_stream(500, 2500, 1.2, seed=1)
    runs = []
    for admission in (False, True):
        clock = Clock()
        cache = make_cache((100, 1000, 10000), admission, clock)
        hits = [request(cache, q, clock) for q in stream]
        runs.append((hits, placement(cache)))
    assert runs[0] == runs[1]


def test_full_cache_rejects_one_time_question():
    cache = make_cache((1, 1, 1), admission=True)
    for q in ["p1", "p2", "p3"]:
        for _ in range(3):
            request(cache, q)
    assert request(cache, "once") is False
    assert cache.rejected_admissions == 1
    assert all(request(cache, q) for q in ["p1", "p2", "p3"])  # popular answers retained


def test_more_requested_question_displaces_least_requested_entry():
    cache = make_cache((3, 0, 0), admission=True)  # single tier: every entry can leave
    for q, n in [("a", 3), ("b", 3), ("rare", 1)]:
        for _ in range(n):
            request(cache, q)
    for _ in range(2):
        request(cache, "new")  # first request rejected, second admitted (count 2 > 1)
    keys = {k for t in cache.tiers for k in t}
    assert hashlib.sha256(b"new").hexdigest() in keys
    assert hashlib.sha256(b"rare").hexdigest() not in keys


def test_without_filter_every_answer_is_stored():
    cache = make_cache((1, 1, 1), admission=False)
    for q in ["p1", "p1", "p2", "p3", "once"]:
        request(cache, q)
    assert cache.rejected_admissions == 0
    assert hashlib.sha256(b"once").hexdigest() in {k for t in cache.tiers for k in t}
