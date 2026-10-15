"""Counts, on the seed-1 main stream of each dataset, how often HMAG runs the verifier and how
often its claim graph reports a contradiction."""
import time
from collections import Counter

import run_revision_experiments as R
from hmag_benchmark import get_workload
from hmag_core import QueryContext
from hmag_retrieval import HierarchicalRetriever
from hmag_system import HMAGConfig, HMAGSystem

shared = R._shared_models(R.PRIMARY_MODEL)
for ds in R.DATASETS:
    wl = get_workload(ds, 500, 1)
    retriever = HierarchicalRetriever(use_gpu=True)
    retriever.add_documents(wl.corpus)
    system = HMAGSystem(HMAGConfig(ollama_model=R.PRIMARY_MODEL), retriever=retriever, **shared)
    counts = Counter()
    for wq in wl.queries:
        r = system.process_query(wq.query, context=QueryContext(query=wq.query, timestamp=time.time()))
        cr = r.consistency_result
        if cr is None:
            continue
        counts['verified'] += 1
        types = {c['type'] for c in cr.contradictions}
        counts['inconsistent'] += int(not cr.is_consistent)
        for t in types:
            counts[t] += 1
        counts['resolution_' + cr.resolution_strategy] += 1
    print(ds, dict(counts), flush=True)
