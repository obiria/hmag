"""Consistency verifier (Algorithm 4) and the sentence-level factual-consistency metric.

The verifier compares cached with freshly retrieved documents using a pretrained NLI
cross-encoder in both directions, embedding similarity, a claim graph built from
subject-predicate-object triples, and a check of dates stated for the same subject."""

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import spacy
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer, util

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyResult:
    is_consistent: bool
    confidence_score: float
    contradictions: List[Dict[str, Any]]
    supporting_evidence: List[Dict[str, Any]]
    resolution_strategy: str
    latency_ms: float


@dataclass
class FactualClaim:
    text: str
    subject: Optional[str]
    predicate: Optional[str]
    object: Optional[str]
    confidence: float
    source_id: str
    timestamp: float


class NLIConsistencyChecker:
    """Pretrained NLI cross-encoder (cross-encoder/nli-deberta-v3-small)."""

    LABELS = ["contradiction", "entailment", "neutral"]  # output order of the model

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small",
                 model: Optional[CrossEncoder] = None):
        self.model = model or CrossEncoder(model_name)

    def predict(self, premises: List[str], hypotheses: List[str]) -> np.ndarray:
        """(N, 3) array of [contradiction, entailment, neutral] probabilities."""
        if not premises:
            return np.zeros((0, 3))
        return self.model.predict(list(zip(premises, hypotheses)), apply_softmax=True, convert_to_numpy=True)

    def bidirectional_score_batch(self, doc_pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        """Scores each pair in both directions. Per pair: the larger contradiction
        probability, the mean entailment probability and the agreement
        1 - max contradiction."""
        if not doc_pairs:
            return []
        premises, hypotheses = [], []
        for d1, d2 in doc_pairs:
            premises += [d1, d2]
            hypotheses += [d2, d1]
        probs = self.predict(premises, hypotheses)
        results = []
        for i in range(len(doc_pairs)):
            pair = probs[2 * i:2 * i + 2]
            contradiction = float(np.max(pair[:, 0]))
            results.append({'contradiction_score': contradiction,
                            'entailment_score': float(np.mean(pair[:, 1])),
                            'agreement': 1.0 - contradiction})
        return results


class FactExtractor:
    """Extracts subject-predicate-object triples (spaCy dependency parse) and numeric
    and date mentions from a text."""

    PATTERNS = [
        r"\b\d+(?:\.\d+)?(?:\s+(?:percent|%|million|billion|thousand))?\b",
        r"\b(?:\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|January|February|March|April|May|June|July|August"
        r"|September|October|November|December)\b",
    ]

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract_claims(self, text: str, source_id: str) -> List[FactualClaim]:
        claims = []
        for sent in self.nlp(text).sents:
            for token in sent:
                if token.dep_ != "ROOT":
                    continue
                subject = obj = None
                for child in token.children:
                    if child.dep_ == "nsubj":
                        subject = child.text
                    elif child.dep_ in ("dobj", "attr", "prep"):
                        obj = child.text
                if subject and obj:
                    claims.append(FactualClaim(text=sent.text, subject=subject, predicate=token.text,
                                               object=obj, confidence=0.8, source_id=source_id,
                                               timestamp=time.time()))
        for pattern in self.PATTERNS:
            for match in re.findall(pattern, text):
                claims.append(FactualClaim(text=match, subject=None, predicate="contains", object=match,
                                           confidence=0.9, source_id=source_id, timestamp=time.time()))
        return claims


class KnowledgeGraph:
    """Claim graph of one verification: entities are nodes and every claim is an edge
    labelled with its predicate. Two claims that connect the same subject and object with
    opposite predicates are reported as a contradiction."""

    OPPOSITE_PREDICATES = [("is", "is not"), ("has", "lacks"), ("increases", "decreases"), ("before", "after")]

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def reset(self):
        self.graph = nx.MultiDiGraph()

    def add_claims(self, claims: List[FactualClaim]):
        for claim in claims:
            if claim.subject and claim.object:
                self.graph.add_edge(claim.subject, claim.object, predicate=claim.predicate,
                                    confidence=claim.confidence, source_id=claim.source_id)

    def find_contradictions(self) -> List[Dict[str, Any]]:
        contradictions = []
        for node1, node2 in set(self.graph.edges()):
            by_predicate = defaultdict(list)
            for edge in self.graph.get_edge_data(node1, node2).values():
                by_predicate[edge['predicate']].append(edge)
            for pred1, pred2 in self.OPPOSITE_PREDICATES:
                if pred1 in by_predicate and pred2 in by_predicate:
                    contradictions.append({
                        'node1': node1, 'node2': node2, 'predicate1': pred1, 'predicate2': pred2,
                        'sources': [by_predicate[pred1][0]['source_id'], by_predicate[pred2][0]['source_id']],
                    })
        return contradictions


class AdvancedVerifier:
    """Consistency verifier of Algorithm 4."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
                 consistency_threshold: float = 0.85,
                 contradiction_threshold: float = 0.7,
                 semantic_similarity_threshold: float = 0.9,
                 encoder: Optional[SentenceTransformer] = None,
                 nli_model: Optional[CrossEncoder] = None,
                 mode: str = "full",
                 fact_extractor: Optional[FactExtractor] = None):
        """`mode` "full" runs all checks; "nli_only" uses the NLI and similarity checks
        only (the NLI-based verifier of Table 7). Encoder, NLI model and fact extractor
        may be shared between instances."""
        if mode not in ("full", "nli_only"):
            raise ValueError(f"Unknown verifier mode: {mode}")
        self.mode = mode
        self.encoder = encoder or SentenceTransformer(model_name)
        self.nli_checker = NLIConsistencyChecker(nli_model_name, model=nli_model)
        self.fact_extractor = fact_extractor or FactExtractor()
        self.knowledge_graph = KnowledgeGraph()
        self.consistency_threshold = consistency_threshold
        self.contradiction_threshold = contradiction_threshold
        self.semantic_similarity_threshold = semantic_similarity_threshold

    def verify_consistency(self, cached_docs: List[str], retrieved_docs: List[str], query: str) -> ConsistencyResult:
        start_time = time.perf_counter()
        neural_results = self._neural_consistency_check(cached_docs, retrieved_docs)
        semantic_results = self._semantic_similarity_check(cached_docs, retrieved_docs)

        if self.mode == "nli_only":
            kg_contradictions, temporal_results, supporting = [], {'temporal_conflicts': []}, []
        else:
            cached_claims = [c for i, d in enumerate(cached_docs)
                             for c in self.fact_extractor.extract_claims(d, f"cache_{i}")]
            retrieved_claims = [c for i, d in enumerate(retrieved_docs)
                                for c in self.fact_extractor.extract_claims(d, f"retrieval_{i}")]
            self.knowledge_graph.reset()
            self.knowledge_graph.add_claims(cached_claims + retrieved_claims)
            kg_contradictions = self.knowledge_graph.find_contradictions()
            temporal_results = self._temporal_consistency_check(cached_claims, retrieved_claims)
            supporting = self._find_supporting_evidence(cached_claims, retrieved_claims)

        is_consistent, confidence, contradictions = self._aggregate_results(
            kg_contradictions, neural_results, semantic_results, temporal_results)
        return ConsistencyResult(
            is_consistent=is_consistent,
            confidence_score=confidence,
            contradictions=contradictions,
            supporting_evidence=supporting,
            resolution_strategy=self._determine_resolution_strategy(is_consistent, confidence, contradictions),
            latency_ms=(time.perf_counter() - start_time) * 1000,
        )

    def _neural_consistency_check(self, cached_docs: List[str], retrieved_docs: List[str]) -> Dict:
        """NLI over all cached-retrieved document pairs."""
        results = {'consistency_scores': [], 'confidence_scores': [], 'contradictions': []}
        pairs = [(c, r) for c in cached_docs for r in retrieved_docs]
        for (cached_doc, retrieved_doc), scores in zip(pairs, self.nli_checker.bidirectional_score_batch(pairs)):
            if scores['contradiction_score'] > self.contradiction_threshold:
                results['contradictions'].append({'cached_doc': cached_doc[:200],
                                                  'retrieved_doc': retrieved_doc[:200],
                                                  'contradiction_score': scores['contradiction_score']})
            results['consistency_scores'].append(scores['entailment_score'])
            results['confidence_scores'].append(scores['agreement'])
        return results

    def _semantic_similarity_check(self, cached_docs: List[str], retrieved_docs: List[str]) -> Dict:
        sim = util.pytorch_cos_sim(torch.FloatTensor(self.encoder.encode(cached_docs, convert_to_numpy=True)),
                                   torch.FloatTensor(self.encoder.encode(retrieved_docs, convert_to_numpy=True)))
        return {'avg_similarity': sim.mean().item(), 'max_similarity': sim.max().item(),
                'min_similarity': sim.min().item()}

    def _temporal_consistency_check(self, cached_claims: List[FactualClaim],
                                    retrieved_claims: List[FactualClaim]) -> Dict:
        """Flags temporal claims about the same subject that state different values.
        Claims without a subject are not compared."""
        cached_temporal = [c for c in cached_claims if self._is_temporal_fact(c)]
        retrieved_temporal = [c for c in retrieved_claims if self._is_temporal_fact(c)]
        conflicts = [
            {'subject': c.subject, 'cached_time': c.object, 'retrieved_time': r.object,
             'conflict_type': 'temporal_mismatch'}
            for c in cached_temporal for r in retrieved_temporal
            if c.subject and c.subject == r.subject and c.object and r.object and c.object != r.object
        ]
        return {'temporal_conflicts': conflicts, 'num_temporal_facts': len(cached_temporal) + len(retrieved_temporal)}

    @staticmethod
    def _is_temporal_fact(claim: FactualClaim) -> bool:
        text = claim.text.lower() if claim.text else ""
        if any(k in text for k in ('year', 'month', 'day', 'date', 'time', 'when', 'before', 'after')):
            return True
        return bool(re.search(r'\d{4}', text) or re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', text))

    def _aggregate_results(self, kg_contradictions: List, neural_results: Dict, semantic_results: Dict,
                           temporal_results: Dict) -> Tuple[bool, float, List]:
        """Consistent when no check reports a contradiction and the mean of the pairwise
        agreements and the mean embedding similarity exceeds the consistency threshold."""
        contradictions = ([{'type': 'knowledge_graph', 'details': c} for c in kg_contradictions]
                          + [{'type': 'neural', 'details': c} for c in neural_results.get('contradictions', [])]
                          + [{'type': 'temporal', 'details': c} for c in temporal_results.get('temporal_conflicts', [])])
        scores = list(neural_results.get('confidence_scores', []))
        if semantic_results.get('avg_similarity') is not None:
            scores.append(semantic_results['avg_similarity'])
        confidence = float(np.mean(scores)) if scores else 0.5
        is_consistent = not contradictions and confidence > self.consistency_threshold
        return is_consistent, confidence, contradictions

    @staticmethod
    def _determine_resolution_strategy(is_consistent: bool, confidence: float, contradictions: List) -> str:
        if is_consistent:
            return "use_hybrid"
        if confidence < 0.3:
            return "full_retrieval"
        if len(contradictions) > 5:
            return "retrieval_with_verification"
        types = {c['type'] for c in contradictions}
        if 'temporal' in types:
            return "prefer_recent"
        if 'knowledge_graph' in types:
            return "fact_verification"  # recorded; the context is left unchanged
        return "weighted_merge"

    def _find_supporting_evidence(self, cached_claims: List[FactualClaim],
                                  retrieved_claims: List[FactualClaim]) -> List[Dict]:
        """Pairs of claims with the same subject whose embeddings have cosine similarity
        above 0.8."""
        pairs = [(c, r) for c in cached_claims for r in retrieved_claims
                 if c.text and r.text and c.subject == r.subject]
        if not pairs:
            return []
        texts = list({c.text for c, _ in pairs} | {r.text for _, r in pairs})
        emb = self.encoder.encode(texts, convert_to_numpy=True)
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
        vec = dict(zip(texts, emb))
        return [{'cached_claim': c.text, 'retrieved_claim': r.text, 'confidence': (c.confidence + r.confidence) / 2}
                for c, r in pairs if float(np.dot(vec[c.text], vec[r.text])) > 0.8]


@dataclass
class FactualConsistencyScore:
    """Shares of response sentences entailed, contradicted or neither by the context."""
    entailed_fraction: float
    contradicted_fraction: float
    unsupported_fraction: float
    num_sentences: int


class FactualConsistencyEvaluator:
    """Sentence-level factual consistency: each sentence of a response is compared by NLI
    with every context document and counts as entailed if some document entails it with
    probability at least `entail_threshold`, otherwise as contradicted if some document
    contradicts it with probability at least `contradict_threshold`. Entailment is tested
    first because the NLI model also assigns high contradiction probabilities to unrelated
    premises."""

    # Leading attributions such as "According to Document 2," are removed before scoring.
    _CITATION_PREAMBLE_RE = re.compile(
        r'^(according to|as (?:per|stated in|reported in|mentioned in)|per|based on)\s+'
        r'(document\s*\d*|the (?:document|context|source|report|passage)s?)\s*[:,]?\s*',
        re.IGNORECASE,
    )

    def __init__(self, nli_checker: Optional[NLIConsistencyChecker] = None,
                 entail_threshold: float = 0.5, contradict_threshold: float = 0.5):
        self.nli_checker = nli_checker or NLIConsistencyChecker()
        self.entail_threshold = entail_threshold
        self.contradict_threshold = contradict_threshold
        try:
            self._nlp = spacy.load("en_core_web_sm")
        except OSError:
            self._nlp = None

    def _split_sentences(self, text: str) -> List[str]:
        if self._nlp is not None:
            return [s.text.strip() for s in self._nlp(text).sents if s.text.strip()]
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    def _strip_citation_preamble(self, sentence: str) -> str:
        stripped = self._CITATION_PREAMBLE_RE.sub('', sentence).strip()
        return stripped[0].upper() + stripped[1:] if stripped else sentence

    def score(self, response: str, context_docs: List[str]) -> FactualConsistencyScore:
        sentences = self._split_sentences(response)
        if not sentences or not context_docs:
            return FactualConsistencyScore(0.0, 0.0, 1.0, len(sentences))
        entailed = contradicted = 0
        for sent in sentences:
            sent = self._strip_citation_preamble(sent)
            best_entail = best_contra = 0.0
            for doc in context_docs:
                probs = self.nli_checker.predict([doc], [sent])[0]
                best_contra = max(best_contra, float(probs[0]))
                best_entail = max(best_entail, float(probs[1]))
            if best_entail >= self.entail_threshold:
                entailed += 1
            elif best_contra >= self.contradict_threshold:
                contradicted += 1
        n = len(sentences)
        return FactualConsistencyScore(entailed / n, contradicted / n, (n - entailed - contradicted) / n, n)
