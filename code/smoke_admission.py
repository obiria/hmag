"""End-to-end check of the cache admission filter: 60 SQuAD v2.0 queries through HMAG
with a five-entry answer cache, so that the cache is full and the filter acts."""
import sys

import run_revision_experiments as R
from hmag_benchmark import get_workload
from hmag_retrieval import HierarchicalRetriever
from hmag_system import HMAGConfig, HMAGSystem

wl = get_workload("squad_v2", 500, 1)
retriever = HierarchicalRetriever(use_gpu=True)
retriever.add_documents(wl.corpus)
cfg = HMAGConfig(ollama_model=R.PRIMARY_MODEL, l1_cache_size=1, l2_cache_size=2, l3_cache_size=2)
system = HMAGSystem(cfg, retriever=retriever, **R._shared_models(R.PRIMARY_MODEL))
hits = errors = 0
for wq in wl.queries[:60]:
    r = system.process_query(wq.query)
    hits += r.cache_hit
    errors += bool(r.metadata.get("generation_error"))
stored = sum(len(t) for t in system.cache.tiers)
print(f"hits={hits} errors={errors} stored={stored} rejected={system.cache.rejected_admissions}")
ok = errors == 0 and stored <= 5 and system.cache.rejected_admissions > 0 and hits > 0
print("ADMISSION SMOKE " + ("PASSED" if ok else "FAILED"))
sys.exit(0 if ok else 1)
