"""
Configurations of the component ablations and the capacity-pressure experiment.

Each variant changes HMAGConfig fields of the full system and otherwise runs the same
code as HMAG. run_revision_experiments.py and run_stress.py run the configurations on
identical query streams.

  full_system                  HMAG with the Thompson-sampling controller and all stages enabled
  no_hierarchical_cache        a single tier of the same total capacity (Adaptive-Flat)
  no_adaptive_controller       the Hybrid strategy for every query instead of the learned controller
  no_verifier                  consistency verification disabled
  no_generation                cache, controller, retrieval and verification only, no LLM call
  policy_gradient_controller   the REINFORCE controller instead of Thompson sampling
  *_strategy                   one fixed retrieval strategy for every query
  retrieval_only_strategy      Retrieval-only for every query without the semantic answer
                               lookup, so that only verbatim repeats are reused
                               ("w/o semantic matching")
  hierarchical_lru / _lfu      three tiers whose priority is recency / access frequency only
  hmag_fixed_temp              priority computed once at insertion and never updated
"""

from typing import Dict

from hmag_core import RetrievalStrategy

STRATEGY_INDEX = {s: i for i, s in enumerate(RetrievalStrategy)}


def _fixed(strategy: RetrievalStrategy) -> Dict:
    return {"controller_type": "fixed", "fixed_strategy_idx": STRATEGY_INDEX[strategy]}


ABLATION_CONFIGS: Dict[str, Dict] = {
    "full_system": {},
    "no_hierarchical_cache": {"l2_cache_size": 0, "l3_cache_size": 0, "l1_cache_size": 11100},
    "no_adaptive_controller": _fixed(RetrievalStrategy.HYBRID),
    "no_verifier": {"enable_verification": False},
    "no_generation": {"enable_generation": False},
    "policy_gradient_controller": {"controller_type": "policy_gradient"},
    "cache_only_strategy": _fixed(RetrievalStrategy.CACHE_ONLY),
    "retrieval_only_strategy": {**_fixed(RetrievalStrategy.RETRIEVAL_ONLY), "semantic_answer_lookup": False},
    "adaptive_hybrid_strategy": _fixed(RetrievalStrategy.ADAPTIVE_HYBRID),
    "fallback_strategy": _fixed(RetrievalStrategy.FALLBACK),
    "hierarchical_strategy": _fixed(RetrievalStrategy.HIERARCHICAL),
    "hierarchical_lru": {"cache_policy": "lru"},
    "hierarchical_lfu": {"cache_policy": "lfu"},
    "hmag_fixed_temp": {"cache_policy": "fixed_temp"},
    # HMAG (original placement): new answers in L1, promotion only into free space
    "hmag_original_placement": {"cache_placement": "insert_l1", "cache_frequency_norm": "exp",
                                "promote_threshold": 0.7, "cache_promote_to_top": False,
                                "cache_promote_sweep": True},
}
