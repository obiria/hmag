# Capacity-pressure experiment

Compares how many useful answers HMAG's cache retains with those of other retention
policies when the number of distinct questions far exceeds the cache capacity.

Workload
- Datasets: SQuAD v2.0, Natural Questions, MS MARCO, with the corpora of the main
  comparison.
- One stream of 2,000 queries per dataset, seed 1, Zipf exponent alpha = 0.8.
- Question pools of five times the stream length, capped by the available questions:
  10,000 questions (SQuAD v2.0, MS MARCO) and 3,771 (Natural Questions).
- The first 10% of each stream is warm-up.

Configurations (total capacity 100 answers for every method)
- HMAG: tiers 10 / 30 / 60, priority score, admission test, Thompson-sampling
  controller, verifier.
- HMAG w/o admission: HMAG without the admission test.
- Hierarchical-LRU, Hierarchical-LFU, HMAG-FixedTemp: same tiers, priority replaced by
  recency, frequency, or the priority computed once at insertion; no admission test.
- Adaptive-Flat: one tier of 100 entries, same controller and verifier; no admission test.
- HMAG (original placement): admission test, new answers in L1, promotion only into
  free space, exponential frequency term, promotion threshold 0.7, periodic sweep.
- SCALM and MinCache: their own replacement policies with capacity 100.

Generation: the short-answer system prompt of hmag_qa_metrics.py, identical for all.

Analysis
- Cache hit rate and mean latency, each configuration compared with HMAG on the same
  stream: exact McNemar test (hits) and Wilcoxon signed-rank test (latency), Holm
  correction across the comparisons within each dataset.
- Question-level token F1, exact match and ROUGE-L; exact-question hit rate against the
  bound of Theorem 1 (0.300 for SQuAD v2.0 and MS MARCO, 0.378 for Natural Questions
  at C = 100).
