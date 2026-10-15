"""Pre-run check of HMAG on 60 SQuAD v2.0 queries: no failed generations, prompts
within the context window and within the document budget of the chosen strategy."""
import sys
import time

import run_revision_experiments as R
from hmag_benchmark import get_workload
from hmag_core import QueryContext
from hmag_retrieval import HierarchicalRetriever
from hmag_system import HMAGConfig, HMAGSystem

wl = get_workload("squad_v2", 500, 1)
retriever = HierarchicalRetriever(use_gpu=True)
retriever.add_documents(wl.corpus)
cfg = HMAGConfig(ollama_model=R.PRIMARY_MODEL)
system = HMAGSystem(cfg, retriever=retriever, **R._shared_models(R.PRIMARY_MODEL))
problems = []
for wq in wl.queries[:60]:
    r = system.process_query(wq.query, context=QueryContext(query=wq.query, timestamp=time.time()))
    if r.metadata.get("generation_error"):
        problems.append("generation error")
    if r.metadata.get("prompt_tokens", 0) >= 4096:
        problems.append(f"prompt of {r.metadata['prompt_tokens']} tokens")
    # Fallback may use more than the usual number of documents
    budget = cfg.retrieval_top_k * (2 if r.strategy_used.name == "FALLBACK" else 1)
    if len(r.documents_used) > budget:
        problems.append(f"{len(r.documents_used)} documents in prompt ({r.strategy_used.name})")
print("smoke:", "OK" if not problems else problems[:5])
sys.exit(1 if problems else 0)
