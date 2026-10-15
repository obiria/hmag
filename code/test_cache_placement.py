"""Tier placement of HierarchicalCache: promotion into a full tier, downward movement of
answers whose popularity declines, and the insert-in-L1 placement."""
import hashlib

import numpy as np
import pytest

from hmag_core import HierarchicalCache


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make(placement, frequency_norm="exp", tiers=(1, 2, 4), promote=0.7):
    rng = np.random.default_rng(0)
    vecs = {}
    clock = Clock()
    c = HierarchicalCache(*tiers, embedding_dim=8, encoder=object(), clock=clock, admission_filter=True,
                          promote_threshold=promote, placement=placement, frequency_norm=frequency_norm,
                          recency_lambda=1e-2)

    def enc(text):
        if text not in vecs:
            v = rng.standard_normal(8).astype("float32")
            vecs[text] = v / np.linalg.norm(v)
        return vecs[text]

    c.encode_cached = enc
    return c, clock


def key(q):
    return hashlib.sha256(q.encode()).hexdigest()


def request(c, clock, q):
    """Request sequence of HMAGSystem.process_query: exact lookup, otherwise store."""
    clock.now += 1.0
    if c.get(key(q)) is None:
        c.search(q, k=5, entry_type_filter="qa_response")
        c.put(key=key(q), content=q, embedding_text=q, entry_type="qa_response")
    c.maintain()  # promotions are applied after the answer is returned


@pytest.mark.parametrize("norm", ["exp", "log"])
def test_hot_answer_reaches_full_l1(norm):
    c, clock = make("hot", norm)
    for q in ["a", "b", "c", "d", "e", "f", "g", "h"]:
        request(c, clock, q)
    assert len(c.tiers[0]) == 1 and len(c.tiers[1]) == 2 and c.tier_of(key("c")) == 2  # tiers full, "c" in L3
    target = "c"
    for _ in range(8):
        request(c, clock, target)
    assert c.tier_of(key(target)) == 0


@pytest.mark.parametrize("norm", ["exp", "log"])
def test_cooled_answer_moves_down(norm):
    c, clock = make("hot", norm)
    for q in ["a", "b", "c", "d", "e", "f", "g", "h"]:
        request(c, clock, q)
    for _ in range(8):
        request(c, clock, "c")
    assert c.tier_of(key("c")) == 0
    clock.now += 2000.0  # "c" goes cold
    for _ in range(12):
        request(c, clock, "e")
    assert c.tier_of(key("e")) == 0
    assert c.tier_of(key("c")) in (1, 2)


def test_insert_l1_placement():
    c, clock = make("insert_l1")
    for q in ["a", "b", "c"]:
        request(c, clock, q)
    assert c.tier_of(key("c")) == 0  # a new answer enters L1


@pytest.mark.parametrize("placement", ["insert_l1", "hot"])
def test_directory_matches_per_tier_search(placement):
    """With every list probed, the directory finds the same hits and tiers as the per-tier
    search, and evicted answers leave the directory."""
    rng = np.random.default_rng(3)
    base = rng.standard_normal((60, 8)).astype("float32")
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    texts = [f"q{i}" for i in range(60)]
    vec = {t: v for t, v in zip(texts, base)}
    for i in range(20):  # near-duplicates that match their source semantically
        v = base[i] + 0.05 * rng.standard_normal(8).astype("float32")
        vec[f"p{i}"] = v / np.linalg.norm(v)
    caches = []
    for idx in ("per_tier", "directory"):
        clock = Clock()
        c = HierarchicalCache(2, 4, 8, embedding_dim=8, encoder=object(), clock=clock, admission_filter=True,
                              placement=placement, search_index=idx, directory_train=base,
                              directory_nprobe=10_000, similarity_thresholds=(0.9, 0.9, 0.9))
        c.encode_cached = lambda t: vec[t]
        caches.append((c, clock))
    order = np.random.default_rng(4).choice(list(vec), 300)
    results = []
    for c, clock in caches:
        out = []
        for q in order:
            clock.now += 1.0
            k = key(q)
            if c.get(k) is not None:
                out.append(("exact", c.tier_of(k)))
            else:
                _, h, t = c.search(q, k=5, entry_type_filter="qa_response")
                out.append(("semantic", t) if h else ("miss", None))
                c.put(key=k, content=q, embedding_text=q, entry_type="qa_response")
            c.maintain()
        results.append(out)
        if c.directory is not None:
            assert set(c._dir_id) == {k for tier in c.tiers for k in tier}
    assert results[0] == results[1]


def test_slot_index_matches_brute_force():
    from hmag_core import SlotIndex
    rng = np.random.default_rng(5)
    idx, live = SlotIndex(16), {}
    for step in range(2000):
        if live and rng.random() < 0.4:
            k = list(live)[int(rng.integers(len(live)))]
            idx.remove(k)
            del live[k]
        else:
            v = rng.standard_normal(16).astype("float32")
            v /= np.linalg.norm(v)
            live[f"k{step}"] = v
            idx.add(f"k{step}", v)
        if step % 50 == 0 and live:
            q = rng.standard_normal(16).astype("float32")
            q /= np.linalg.norm(q)
            got = [k for k, s in idx.search(q, 5) if s > 0]
            ref = sorted(live, key=lambda k: -float(live[k] @ q))[:5]
            ref = [k for k in ref if float(live[k] @ q) > 0]
            assert got == ref
