"""SNLI validation pairs for evaluating the verifier's NLI model; the rest of the
verifier evaluation is in run_revision_experiments.stage_verifier."""

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

NLI_LABELS = ["contradiction", "entailment", "neutral"]


def _load_nli_eval_set(n: int = 300, seed: int = 42) -> List[Tuple[str, str, str]]:
    """Returns `n` random (premise, hypothesis, label) triples of the SNLI validation
    split, excluding pairs without a gold label."""
    try:
        from datasets import load_dataset
        ds = load_dataset("stanfordnlp/snli", split="validation")
        ds = ds.filter(lambda ex: ex['label'] in (0, 1, 2))
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
        # SNLI label ids: 0=entailment, 1=neutral, 2=contradiction
        id_to_label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        return [(ds[int(i)]['premise'], ds[int(i)]['hypothesis'], id_to_label[ds[int(i)]['label']])
                for i in idx]
    except Exception as e:
        raise RuntimeError(f"SNLI could not be loaded ({e}); no substitute data is used.") from e
