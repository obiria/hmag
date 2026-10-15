"""
Parameter grid of the one-factor-at-a-time sensitivity analysis.

Each parameter is varied over five values while all others keep their
defaults; run_revision_experiments.stage_sensitivity runs every setting on the
same SQuAD v2.0 stream.
"""

from typing import Any, Dict, List

DEFAULT_SWEEP_GRID: Dict[str, List[Any]] = {
    "similarity_threshold": [0.70, 0.80, 0.85, 0.90, 0.95],
    "promote_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    "demote_threshold": [0.1, 0.2, 0.3, 0.4, 0.5],
    # noise scale sigma of the Thompson-sampling controller (sets its exploration)
    "thompson_sigma": [0.25, 0.5, 1.0, 2.0, 4.0],
    "consistency_threshold": [0.70, 0.80, 0.85, 0.90, 0.95],
    "contradiction_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
}
