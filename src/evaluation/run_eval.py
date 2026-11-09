"""
src/evaluation/run_eval.py - End-to-end RAG evaluation script with configurable experiments.

BUSINESS PROBLEM: Runs reproducible, configurable end-to-end evaluations of the
telecom RAG pipeline (single question, golden set, ablation study, configuration
comparison) and writes results to markdown/JSON. It is the harness that answers
"does this chunking/retrieval/reranker change actually improve QA quality?"

WHERE IT SITS IN THE DATA FLOW: At the end of the flow, after documents are
ingested and the pipeline is wired. It ingests/indexes documents itself, runs
retrieve -> (rerank) -> generate per question, scores with RAGEvaluator, and
writes reports. It does not serve live traffic.

WHAT CALLS IT: python -m src.evaluation.run_eval (CLI via argparse), CI jobs,
and developers comparing pipeline configurations.

WHAT IT CALLS: src.ingestion.* (loaders, cleaner, chunker), src.embedding.embedder,
src.store.vector_store, src.retrieval.* (retriever, reranker, hybrid, sparse_retriever,
reciprocal_rank_fusion, _doc_key), src.generation.* (answer, prompt),
src.evaluation.metrics (RAGEvaluator, RetrievalMetrics, GenerationMetrics),
providers.llm.create_llm_provider_from_env.

WHAT BREAKS IF REMOVED: There is no scripted way to run golden-set or ablation
evaluations; pipeline changes cannot be measured before deploy.
"""

import json
import os
import sys
from typing import List, Dict, Any, Set, Optional, Tuple
from pathlib import Path
import itertools

# Add project root to path for imports
# Allows `src.*` imports when this file is run directly as a script.
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.ingestion.loaders import load_documents_from_directory
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_documents
from src.embedding.embedder import embed_documents
from src.store.vector_store import create_vector_store, add_documents_to_store
from src.retrieval.retriever import create_retriever, create_hybrid_retriever
from src.retrieval.reranker import create_reranker
from src.retrieval.hybrid import create_hybrid_retriever_rrf
from src.generation.answer import create_answer_generator
from src.generation.prompt import create_grounded_prompt, format_context, PromptEngineer
from src.evaluation.metrics import RAGEvaluator, RetrievalMetrics, GenerationMetrics
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_llm_judge():
    """Build an LLM-as-judge function from the env-selected provider.

    Returns a callable(prompt) -> str. With LLM_PROVIDER=mock this is the mock
    (CI-safe, no key); with nvidia it is the real NVIDIA model. Judge limitations
    are documented in learn/04-evaluation.md — it is not ground truth.

    WHAT: Factory that creates a judge(prompt)->str closure backed by the
    provider selected via LLM_PROVIDER env var.
    WHY: Supplies the LLM judge used by GenerationMetrics for relevance,
    faithfulness, and contradiction scoring without hardcoding a provider.
    INPUTS: none (reads LLM_PROVIDER from env inside create_llm_provider_from_env).
    OUTPUTS: a callable judge(prompt: str) -> str.
    SIDE EFFECTS: instantiates an LLM provider (may read env / hit network).
    FAILURE MODES: if provider.generate raises, the closure catches it, logs a
        warning, and returns "0" (a neutral low score) so evaluation does not
        crash on judge failure.
    ALTERNATIVE: inject a fixed judge function directly instead of building one
        from env.
    # GAP: resilience - judge failures degrade to "0" silently; there is no
        retry/backoff and no circuit breaker.
    # BUSINESS: enables LLM-as-judge scoring in CI (mock) and prod (NVIDIA)
        without code changes.
    # INTERVIEW: "Why return '0' on judge error instead of re-raising?" - so a
        transient judge outage does not abort an entire golden-set evaluation;
        the score is just conservatively low.
    """
    from providers.llm import create_llm_provider_from_env

    provider = create_llm_provider_from_env()

    def judge(prompt: str) -> str:
        try:
            return provider.generate(prompt)
        except Exception as e:
            logger.warning(f"judge provider error: {e}")
            return "0"

    return judge


class RAGEvaluationPipeline:
    """End-to-end RAG pipeline for evaluation with configurable components.

    WHAT: Wires together ingestion (load->clean->chunk->embed->index), retrieval
    (dense, optional BM25 hybrid with RRF or weighted fusion), optional reranking,
    and answer generation, then scores each question with RAGEvaluator.
    WHY: Lets experiments toggle chunking/hybrid/rerank/RRF/prompt and measure
    the impact on retrieval and generation metrics for the telecom QA system.
    INPUTS: chunk_strategy, chunk_size, chunk_overlap, use_hybrid, use_reranker,
        use_rrf, hybrid_alpha (0-1), prompt_version, llm_judge_fn optional.
    OUTPUTS: none directly; methods return evaluation result dicts.
    SIDE EFFECTS: creates a vector store and reranker at construction; reads
        filesystem and (via make_llm_judge) may hit network when judging.
    FAILURE MODES: __init__ does not fail if retrieval components fail (those
        are created lazily in ingest_and_index_documents); evaluate_* raises
        RuntimeError if ingest_and_index_documents was not called first.
    ALTERNATIVE: call src.ingestion / src.retrieval / src.generation modules
        directly instead of this orchestrating class.
    # GAP: concurrency - no locking; two threads sharing one pipeline could
        race on self.sparse_retriever / self.retriever initialization.
    # GAP: retries - ingest_and_index_documents does not retry embedding or
        vector-store writes on transient failure.
    # BUSINESS: is the experiment harness that tells FDE learners whether a
        retrieval change actually moves telecom QA quality.
    # INTERVIEW: "Why create the vector store and reranker in __init__ but the
        retriever only at ingest time?" - because the retriever needs the
        embedder, which is only built after documents are embedded.
    """

    def __init__(self,
                 chunk_strategy: str = "recursive",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 use_hybrid: bool = True,
                 use_reranker: bool = True,
                 use_rrf: bool = False,
                 hybrid_alpha: float = 0.7,
                 prompt_version: str = "grounded_v1",
                 llm_judge_fn: Optional[callable] = None):
        """Configure and pre-wire the evaluation pipeline components.

        WHAT: Stores config, creates an empty vector store, a reranker, an
        answer generator, a PromptEngineer, and a RAGEvaluator; leaves
        embedder/retriever/sparse_retriever for lazy init at ingest time.
        WHY: Separates configuration from the expensive ingestion so multiple
        experiments can share the same component factory pattern.
        INPUTS: chunk_strategy="recursive"; chunk_size=500; chunk_overlap=50;
            use_hybrid=True; use_reranker=True; use_rrf=False;
            hybrid_alpha=0.7; prompt_version="grounded_v1"; llm_judge_fn=None.
        OUTPUTS: none (sets instance attributes; is_ready=False).
        SIDE EFFECTS: calls create_vector_store() and create_reranker() and
            create_answer_generator() at construction.
        FAILURE MODES: if create_reranker or create_answer_generator fails at
            construction, __init__ raises before the object exists.
        ALTERNATIVE: lazy-create every component inside ingest_and_index_documents.
        # GAP: validation - hybrid_alpha is not range-checked (0-1); an out-of-
            range alpha silently distorts the weighted fusion.
        # BUSINESS: centralizes the experiment knobs so ablation studies vary
            one thing at a time.
        # INTERVIEW: "Why is is_ready False until ingest runs?" - because the
            retriever and BM25 corpus depend on embedded documents that do not
            exist yet.
        """
        logger.info("Initializing RAG evaluation pipeline...")

        # Store configuration
        self.chunk_strategy = chunk_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_hybrid = use_hybrid
        self.use_reranker = use_reranker
        self.use_rrf = use_rrf
        self.hybrid_alpha = hybrid_alpha
        self.prompt_version = prompt_version
        self.llm_judge_fn = llm_judge_fn

        # Initialize components
        self.embedder = None  # Will be initialized when needed (after embed docs)
        self.vector_store = create_vector_store()  # empty vector store to fill at ingest
        self.retriever = None  # Will be initialized after vector store is populated
        self.hybrid_retriever = None  # Will be initialized if using hybrid (unused placeholder)
        self.sparse_retriever = None  # BM25 retriever for hybrid (lazy)
        self.reranker = create_reranker()
        self.answer_generator = create_answer_generator()
        self.prompt_engineer = PromptEngineer()
        self.evaluator = RAGEvaluator()

        # Flag to track if pipeline is ready
        self.is_ready = False

    def ingest_and_index_documents(self, data_directory: str) -> None:
        """Load, clean, chunk, embed, and index documents; build retriever + BM25.

        Args:
            data_directory: Path to directory containing documents

        WHAT: Runs the full ingestion pipeline (load -> clean -> chunk -> embed
        -> index into vector store), builds the dense retriever and optional BM25
        sparse retriever, and marks the pipeline ready.
        WHY: Prepares the search index that evaluate_single_question queries
        against; must run before any evaluation.
        INPUTS: data_directory e.g. "data/raw".
        OUTPUTS: none (sets self.is_ready=True, self._corpus, self.embedder,
            self.retriever, self.sparse_retriever).
        SIDE EFFECTS: reads the filesystem, writes to the vector store, logs at
            each stage.
        FAILURE MODES: raises if document loading/embedding fails; is_ready
            stays False so later evaluate calls raise RuntimeError.
        ALTERNATIVE: incremental indexing (add docs without rebuilding) instead
            of full rebuild each call.
        # GAP: retries - no retry/backoff around load_documents_from_directory
            or embed_documents on transient failure.
        # BUSINESS: produces the searchable telecom document corpus used to
            score retrieval quality.
        # INTERVIEW: "Why rebuild BM25 and dense index from scratch on every
            ingest call?" - simplicity and consistency; incremental adds risk of
            drift between the vector store and the BM25 corpus.
        """
        logger.info(f"Ingesting documents from {data_directory}")
        logger.info(f"Configuration: chunk_strategy={self.chunk_strategy}, "
                   f"chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")

        # Load documents
        raw_docs = load_documents_from_directory(data_directory)
        logger.info(f"Loaded {len(raw_docs)} raw documents")

        # Clean documents
        cleaned_docs = clean_documents(raw_docs)
        logger.info(f"Cleaned {len(cleaned_docs)} documents")

        # Chunk documents
        chunked_docs = chunk_documents(
            cleaned_docs,
            strategy=self.chunk_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        logger.info(f"Created {len(chunked_docs)} chunks using {self.chunk_strategy} strategy")

        # Embed documents
        embedded_docs = embed_documents(chunked_docs)
        logger.info(f"Embedded {len(embedded_docs)} document chunks")

        # Add to vector store
        add_documents_to_store(self.vector_store, embedded_docs)
        logger.info(f"Indexed {len(embedded_docs)} document vectors")

        # Keep the corpus for BM25 (hybrid retrieval) — initialized once here
        self._corpus = embedded_docs

        # Initialize embedder (needed for retriever)
        from src.embedding.embedder import Embedder
        self.embedder = Embedder()

        # Initialize retriever
        self.retriever = create_retriever(self.vector_store, self.embedder)

        # Initialize BM25 over the full corpus when hybrid is enabled
        if self.use_hybrid:
            from src.retrieval.sparse_retriever import create_bm25_retriever
            self.sparse_retriever = create_bm25_retriever()
            self.sparse_retriever.add_documents(embedded_docs)
            logger.info(f"Hybrid retrieval enabled: use_rrf={self.use_rrf}, alpha={self.hybrid_alpha}")

        self.is_ready = True

    def _initialize_sparse_retriever(self, documents: List[Dict[str, Any]]) -> None:
        """Initialize sparse retriever (BM25) with the corpus if not yet done.

        WHAT: Lazily creates a BM25 retriever over the given documents (or the
        stored _corpus) if one does not exist yet.
        WHY: BM25 is only needed for hybrid retrieval, so it is built on demand
        rather than at construction.
        INPUTS: documents list of dicts (may be empty; falls back to self._corpus).
        OUTPUTS: none (sets self.sparse_retriever).
        SIDE EFFECTS: creates and populates the BM25 retriever.
        FAILURE MODES: if self._corpus is also unset and documents is empty,
            BM25 gets no documents and retrieval returns empty.
        ALTERNATIVE: build BM25 eagerly in ingest_and_index_documents regardless
            of use_hybrid.
        # BUSINESS: supports the BM25 half of hybrid retrieval for telecom term
            matching.
        # INTERVIEW: "Why fall back to self._corpus when documents is empty?" -
            so _hybrid_retrieve can call this without re-passing the corpus.
        """
        if self.sparse_retriever is None:
            from src.retrieval.sparse_retriever import create_bm25_retriever
            self.sparse_retriever = create_bm25_retriever()
            self.sparse_retriever.add_documents(documents or getattr(self, "_corpus", []))
            logger.debug("Initialized BM25 retriever")

    def evaluate_single_question(self,
                                question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the RAG pipeline on a single golden-set question.

        Args:
            question_data: Dictionary containing question information

        Returns:
            Evaluation results dictionary

        WHAT: Runs retrieve -> (optional rerank) -> format context -> generate
        answer, derives relevant doc IDs from expected terms, scores with
        RAGEvaluator, and augments the result with question metadata and config.
        WHY: Produces one scored evaluation record per golden-set question.
        INPUTS: question_data dict {"question": str, "id": str,
            "expected_answer_contains": [str], "expected_sources": [str]}.
        OUTPUTS: dict with question_id, question, expected_*, config,
            retrieval_details, plus nested retrieval/generation metrics.
        SIDE EFFECTS: logs progress; invokes the LLM judge (via evaluator) if
            configured.
        FAILURE MODES: raises RuntimeError if is_ready is False; catches nothing
            here - a per-question exception propagates to evaluate_golden_set
            which logs and records an error result.
        ALTERNATIVE: skip the pseudo-relevant fallback and score with empty
            relevant_doc_ids (all zeros).
        # GAP: validation - question_data["question"] is accessed with []
            (KeyError if missing) while other fields use .get(); inconsistent.
        # BUSINESS: the unit of evaluation - one telecom QA question scored end
        to end.
        # INTERVIEW: "Why use top 3 as pseudo-relevant when no expected terms
            match?" - to avoid all-zero scores when the golden set's expected
            terms do not appear in the retrieved chunks; it is a conservative
            fallback so the metric is not degenerate.
        """
        if not self.is_ready:
            raise RuntimeError("Pipeline not ready. Call ingest_and_index_documents first.")

        question = question_data["question"]
        question_id = question_data.get("id", "unknown")
        expected_terms = question_data.get("expected_answer_contains", [])
        expected_sources = question_data.get("expected_sources", [])

        logger.info(f"Evaluating question {question_id}: {question}")

        # Retrieve documents: hybrid when enabled, else dense-only; fetch k=10
        # (reranker will narrow to top-5 afterward).
        if self.use_hybrid:
            retrieved_results = self._hybrid_retrieve(question, k=10)
        else:
            retrieved_results = self.retriever.retrieve(
                question,
                k=10,
                current_version_only=True  # Filter to current versions by default
            )

        retrieved_docs = [doc for doc, _ in retrieved_results]
        retrieval_scores = [score for _, score in retrieved_results]

        # Rerank documents if enabled: cross-encoder re-orders for relevance.
        if self.use_reranker and retrieved_docs:
            reranked_results = self.reranker.rerank(
                question,
                list(zip(retrieved_docs, retrieval_scores)),
                k=5
            )
            final_docs = [doc for doc, _ in reranked_results]
            final_scores = [score for _, score in reranked_results]
        else:
            # Use top k from initial retrieval (no reranking).
            final_docs = retrieved_docs[:5]
            final_scores = retrieval_scores[:5]

        # Format context for generation
        context = format_context(final_docs)

        # Generate answer with the (real or mock) provider via the answer generator
        answer_result = self.answer_generator.generate_answer_with_context(
            question, final_docs)
        generated_answer = answer_result.answer

        # Determine relevant document IDs based on expected terms: a retrieved
        # doc is "relevant" if its content contains any expected term.
        relevant_doc_ids = set()
        expected_terms_lower = [term.lower() for term in expected_terms]

        for i, doc in enumerate(final_docs):
            content_lower = doc["content"].lower()
            # Check if any expected term appears in the document
            if any(term in content_lower for term in expected_terms_lower):
                # Use metadata doc_id if available, otherwise use index
                metadata = doc.get("metadata", {})
                doc_id = metadata.get("doc_id") or metadata.get("filename") or metadata.get("source") or f"doc_{i}"
                relevant_doc_ids.add(doc_id)

        # If no documents matched expected terms but we have results,
        # consider top 3 as relevant for evaluation (to avoid zero scores)
        if not relevant_doc_ids and final_docs:
            for i, doc in enumerate(final_docs[:3]):
                metadata = doc.get("metadata", {})
                doc_id = metadata.get("doc_id") or metadata.get("filename") or metadata.get("source") or f"doc_{i}"
                relevant_doc_ids.add(doc_id)
            logger.debug(f"No expected terms found in retrieved docs for question {question_id}, "
                        f"using top 3 as pseudo-relevant for evaluation")

        # Evaluate the pipeline (retrieval + generation metrics).
        evaluation_result = self.evaluator.evaluate_rag_pipeline(
            query=question,
            retrieved_docs=final_docs,
            generated_answer=generated_answer,
            relevant_doc_ids=relevant_doc_ids,
            expected_keywords=expected_terms,
            llm_judge_fn=self.llm_judge_fn
        )

        # Add question-specific information
        evaluation_result["question_id"] = question_id
        evaluation_result["question"] = question
        evaluation_result["expected_answer_contains"] = expected_terms
        evaluation_result["expected_sources"] = expected_sources
        evaluation_result["config"] = {
            "chunk_strategy": self.chunk_strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "use_hybrid": self.use_hybrid,
            "use_reranker": self.use_reranker,
            "use_rrf": self.use_rrf,
            "hybrid_alpha": self.hybrid_alpha,
            "prompt_version": self.prompt_version
        }
        evaluation_result["retrieval_details"] = {
            "num_initial_retrieved": len(retrieved_docs),
            "num_final_docs": len(final_docs),
            "final_scores": final_scores
        }

        return evaluation_result

    def _hybrid_retrieve(self, query: str, k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """Fuse dense and BM25 results, optionally with Reciprocal Rank Fusion.

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of (document, score) tuples

        WHAT: Retrieves from dense and BM25 (search_k = max(k*3, 10) for a
        candidate pool), then fuses via RRF if use_rrf else min-max normalized
        weighted combination using hybrid_alpha.
        WHY: Combines semantic (dense) and lexical (BM25) signals for better
        retrieval than either alone on telecom terminology.
        INPUTS: query string, k int (default 10).
        OUTPUTS: list of (doc, score) tuples, trimmed to k.
        SIDE EFFECTS: lazily initializes BM25 via _initialize_sparse_retriever;
            logs a warning if BM25 is unavailable (dense-only fallback).
        FAILURE MODES: if BM25 is None, falls back to dense-only results[:k].
        ALTERNATIVE: use only dense or only BM25 (set use_hybrid=False).
        # GAP: validation - hybrid_alpha is not clamped to [0,1]; out-of-range
            values distort the weighted combination silently.
        # BUSINESS: the hybrid retrieval path that ablation studies toggle to
            measure BM25's contribution to telecom chunk matching.
        # INTERVIEW: "Why retrieve k*3 candidates before fusion?" - fusion
            re-ranks, so a wider candidate pool lets the fused ranking surface
            good docs that ranked mediocre in either list alone.
        """
        from src.retrieval.retriever import reciprocal_rank_fusion

        # Ensure BM25 is initialized over the corpus
        if self.sparse_retriever is None:
            self._initialize_sparse_retriever(getattr(self, "_corpus", []))

        search_k = max(k * 3, 10)  # Get extra for fusion to work well
        dense_results = self.retriever.retrieve(
            query, k=search_k, current_version_only=True
        )

        if self.sparse_retriever is None:
            logger.warning("BM25 unavailable - dense only")
            return dense_results[:k]

        sparse_results = self.sparse_retriever.retrieve(query, k=search_k)

        if self.use_rrf:
            fused = reciprocal_rank_fusion([dense_results, sparse_results], k=60)
            return fused[:k]
        else:
            # Weighted score combination (min-max normalize each list to [0,1])
            def norm(results):
                if not results:
                    return results
                scores = [s for _, s in results]
                lo, hi = min(scores), max(scores)
                span = (hi - lo) or 1.0
                return [(d, (s - lo) / span) for d, s in results]

            dense_n = norm(dense_results)
            sparse_n = norm(sparse_results)
            combined = {}
            from src.retrieval.retriever import _doc_key
            for doc, s in dense_n:
                combined.setdefault(_doc_key(doc), [doc, 0.0, 0.0])[1] = s
            for doc, s in sparse_n:
                combined.setdefault(_doc_key(doc), [doc, 0.0, 0.0])[2] = s
            a = self.hybrid_alpha
            out = [(v[0], a * v[1] + (1 - a) * v[2]) for v in combined.values()]
            out.sort(key=lambda x: x[1], reverse=True)
            return out[:k]

    def evaluate_golden_set(self, golden_set_path: str) -> Dict[str, Any]:
        """Evaluate the pipeline on every question in a golden-set JSON file.

        Args:
            golden_set_path: Path to golden set JSON file

        Returns:
            Aggregated evaluation results

        WHAT: Loads the golden set, runs evaluate_single_question per question
        (capturing per-question errors as error records), aggregates metrics,
        and returns individual + aggregate results with config.
        WHY: Produces a reproducible, comparable score across all golden-set
        questions for a given pipeline configuration.
        INPUTS: golden_set_path string pointing to JSON with a "questions" list.
        OUTPUTS: dict {"individual_results": [...], "aggregate_metrics": {...},
            "total_questions": int, "successful_evaluations": int,
            "configuration": {...}}.
        SIDE EFFECTS: reads the golden-set JSON file; logs per-question progress
        and errors.
        FAILURE MODES: a single question's failure is caught and recorded as an
        error record (does not abort the whole set); missing file raises
        FileNotFoundError.
        ALTERNATIVE: fail-fast on the first question error instead of continuing.
        # GAP: resilience - no retry on transient per-question failures (e.g.
            judge timeout); they are recorded as errors and skipped.
        # BUSINESS: the main reproducibility tool for measuring telecom QA
            quality before/after a pipeline change.
        # INTERVIEW: "Why record per-question errors instead of aborting?" -
            so one bad question does not discard scores for all the others;
            the aggregate still reflects the rest of the set.
        """
        logger.info(f"Loading golden set from {golden_set_path}")

        with open(golden_set_path, 'r') as f:
            golden_set_data = json.load(f)

        questions = golden_set_data.get("questions", [])
        logger.info(f"Evaluating {len(questions)} questions from golden set")

        # Evaluate each question
        results = []
        for question_data in questions:
            try:
                result = self.evaluate_single_question(question_data)
                results.append(result)
                logger.info(f"Completed evaluation for question {question_data.get('id', 'unknown')}")
            except Exception as e:
                logger.error(f"Error evaluating question {question_data.get('id', 'unknown')}: {e}")
                # Add error result
                results.append({
                    "question_id": question_data.get("id", "unknown"),
                    "question": question_data["question"],
                    "error": str(e),
                    "config": {
                        "chunk_strategy": self.chunk_strategy,
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "use_hybrid": self.use_hybrid,
                        "use_reranker": self.use_reranker,
                        "use_rrf": self.use_rrf,
                        "hybrid_alpha": self.hybrid_alpha,
                        "prompt_version": self.prompt_version
                    }
                })

        # Calculate aggregate metrics
        aggregated_results = self._aggregate_results(results)

        return {
            "individual_results": results,
            "aggregate_metrics": aggregated_results,
            "total_questions": len(questions),
            "successful_evaluations": len([r for r in results if "error" not in r]),
            "configuration": {
                "chunk_strategy": self.chunk_strategy,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "use_hybrid": self.use_hybrid,
                "use_reranker": self.use_reranker,
                "use_rrf": self.use_rrf,
                "hybrid_alpha": self.hybrid_alpha,
                "prompt_version": self.prompt_version
            }
        }

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Average retrieval and generation metrics across valid question results.

        Args:
            results: List of individual question results

        Returns:
            Dictionary of aggregated metrics

        WHAT: Filters out error records, then averages each retrieval and
        generation metric across the remaining answers (completeness averaged
        only over questions that had expected_keywords).
        WHY: Collapses many per-question scores into one comparable aggregate
        per configuration.
        INPUTS: results list of dicts (may include "error" records).
        OUTPUTS: dict {"retrieval": {...avg...}, "generation": {...avg...},
            "num_questions_evaluated": int} or {"error": ...} if none valid.
        SIDE EFFECTS: none.
        FAILURE MODES: returns an error dict when no valid results exist.
        ALTERNATIVE: report medians or per-question distributions instead of
            means (means hide variance).
        # GAP: observability - no variance/stddev is reported, so a high mean
            could hide a few catastrophic failures.
        # BUSINESS: one comparable number per pipeline configuration for the
            ablation / comparison reports.
        # INTERVIEW: "Why average completeness only over questions that had
            expected_keywords?" - because completeness is undefined (not 0)
            when there are no required keywords; averaging zeros would
            understate it.
        """
        # Filter out results with errors
        valid_results = [r for r in results if "error" not in r]

        if not valid_results:
            return {"error": "No valid results to aggregate"}

        # Initialize aggregators
        retrieval_metrics_sum = {
            "hit_rate_at_5": 0.0,
            "hit_rate_at_10": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "precision_at_5": 0.0,
            "mrr": 0.0,
            "ndcg_at_5": 0.0,
            "average_precision": 0.0
        }

        generation_metrics_sum = {
            "answer_relevance": 0.0,
            "faithfulness": 0.0,
            "contradiction": 0.0,
            "answer_completeness": 0.0
        }

        count = len(valid_results)
        completeness_count = 0

        # Sum metrics
        for result in valid_results:
            retrieval = result.get("retrieval", {})
            generation = result.get("generation", {})

            for key in retrieval_metrics_sum:
                if key in retrieval:
                    retrieval_metrics_sum[key] += retrieval[key]

            for key in generation_metrics_sum:
                if key in generation:
                    generation_metrics_sum[key] += generation[key]
                    if key == "answer_completeness":
                        completeness_count += 1

        # Calculate averages
        retrieval_metrics_avg = {
            key: value / count for key, value in retrieval_metrics_sum.items()
        }

        generation_metrics_avg = {}
        for key, value in generation_metrics_sum.items():
            if key == "answer_completeness":
                generation_metrics_avg[key] = value / completeness_count if completeness_count > 0 else 0.0
            else:
                generation_metrics_avg[key] = value / count

        return {
            "retrieval": retrieval_metrics_avg,
            "generation": generation_metrics_avg,
            "num_questions_evaluated": count
        }

    def run_ablation_study(self, golden_set_path: str, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Compare baseline vs each pipeline improvement on the golden set.

        Args:
            golden_set_path: Path to golden set JSON file
            base_config: Base configuration to compare against

        Returns:
            Ablation study results

        WHAT: Defines several pipeline configurations (baseline, chunking,
        hybrid, hybrid+RRF, reranking, llm-as-judge, advanced prompting),
        overrides each with base_config, runs a full golden-set evaluation for
        each, and returns per-config results.
        WHY: Isolates the contribution of each RAG component to overall quality
        so the team knows what to invest in.
        INPUTS: golden_set_path string, base_config dict of config overrides.
        OUTPUTS: dict {config_name: evaluation_results}.
        SIDE EFFECTS: re-ingests/re-indexes documents for each configuration
            (expensive), logs progress.
        FAILURE MODES: a failing configuration aborts the whole study (no try
            around the per-config loop).
        ALTERNATIVE: share one index across configs where only retrieval knobs
            differ, instead of re-ingesting each time.
        # GAP: resilience - no per-config try/except; one bad config fails the
            entire ablation study.
        # BUSINESS: quantifies which component (chunking/hybrid/rerank/prompt)
            moves telecom QA metrics most.
        # INTERVIEW: "Why re-ingest documents for every configuration?" -
            because chunk_size/strategy change the indexed chunks themselves,
            so the index must be rebuilt; only retrieval-only knobs could share
            an index.
        """
        logger.info("Running ablation study...")

        # Define the configurations to test
        configs = {
            "baseline": {
                "chunk_strategy": "fixed_size",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "use_hybrid": False,
                "use_reranker": False,
                "use_rrf": False,
                "hybrid_alpha": 0.7,
                "prompt_version": "grounded_v1"
            },
            "chunking_improvement": {
                "chunk_strategy": "recursive",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "use_hybrid": False,
                "use_reranker": False,
                "use_rrf": False,
                "hybrid_alpha": 0.7,
                "prompt_version": "grounded_v1"
            },
            "hybrid_retrieval": {
                "chunk_strategy": "recursive",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "use_hybrid": True,
                "use_reranker": False,
                "use_rrf": False,
                "hybrid_alpha": 0.7,
                "prompt_version": "grounded_v1"
            },
            "hybrid_with_rrf": {
                "chunk_strategy": "recursive",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "use_hybrid": True,
                "use_reranker": False,
                "use_rrf": True,
                "hybrid_alpha": 0.7,
                "prompt_version": "grounded_v1"
            },
            "reranking": {
                "chunk_strategy": "recursive",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "use_hybrid": True,
                "use_reranker": True,
                "use_rrf": False,
                "hybrid_alpha": 0.7,
                "prompt_version": "grounded_v1"
            },
            "llm_as_judge": {
                "chunk_strategy": "recursive",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "use_hybrid": True,
                "use_reranker": True,
                "use_rrf": False,
                "hybrid_alpha": 0.7,
                "prompt_version": "grounded_v1"
                # Note: llm_judge_fn would need to be provided separately
            },
            "advanced_prompting": {
                "chunk_strategy": "recursive",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "use_hybrid": True,
                "use_reranker": True,
                "use_rrf": False,
                "hybrid_alpha": 0.7,
                "prompt_version": "grounded_v2"  # Assuming we have a v2
            }
        }

        # Override with base config if provided
        for config_name, config in configs.items():
            config.update(base_config)

        # Run evaluation for each configuration
        ablation_results = {}

        for config_name, config in configs.items():
            logger.info(f"Evaluating configuration: {config_name}")

            # Create new pipeline with this configuration
            pipeline = RAGEvaluationPipeline(
                chunk_strategy=config["chunk_strategy"],
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"],
                use_hybrid=config["use_hybrid"],
                use_reranker=config["use_reranker"],
                use_rrf=config["use_rrf"],
                hybrid_alpha=config["hybrid_alpha"],
                prompt_version=config["prompt_version"],
                llm_judge_fn=self.llm_judge_fn if config_name == "llm_as_judge" else None
            )

            # Ingest and index documents
            data_directory = project_root / "data" / "raw"
            pipeline.ingest_and_index_documents(str(data_directory))

            # Run evaluation
            results = pipeline.evaluate_golden_set(golden_set_path)
            ablation_results[config_name] = results

            logger.info(f"Completed evaluation for {config_name}")

        return ablation_results

def save_results_to_markdown(results: Dict[str, Any], output_path: str) -> None:
    """Write evaluation results (aggregate + first 10 questions) to markdown.

    Args:
        results: Evaluation results dictionary
        output_path: Path to output markdown file

    WHAT: Renders aggregate retrieval/generation metrics, configuration, and the
    first 10 individual question results into a markdown report file.
    WHY: Produces a human-readable report for reviewing experiment outcomes.
    INPUTS: results dict (from evaluate_golden_set or ablation), output_path.
    OUTPUTS: none (writes a file).
    SIDE EFFECTS: creates/overwrites the output markdown file.
    FAILURE MODES: raises IOError if the output path is not writable.
    ALTERNATIVE: emit JSON/HTML instead of markdown.
    # BUSINESS: gives FDE learners a scannable report of which metrics changed.
    # INTERVIEW: "Why only show the first 10 questions?" - to keep the report
        short; full detail remains in the JSON/results dict.
    """
    with open(output_path, 'w') as f:
        f.write("# RAG System Evaluation Results\n\n")

        if "aggregate_metrics" in results:
            agg = results["aggregate_metrics"]
            f.write("## Aggregate Metrics\n\n")

            f.write("### Retrieval Metrics\n")
            for metric, value in agg.get("retrieval", {}).items():
                f.write(f"- {metric}: {value:.3f}\n")
            f.write("\n")

            f.write("### Generation Metrics\n")
            for metric, value in agg.get("generation", {}).items():
                f.write(f"- {metric}: {value:.3f}\n")
            f.write("\n")

            f.write(f"**Questions Evaluated:** {agg.get('num_questions_evaluated', 0)}\n\n")

        if "configuration" in results:
            f.write("## Configuration\n\n")
            config = results["configuration"]
            for key, value in config.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")

        if "individual_results" in results:
            f.write("## Individual Question Results\n\n")
            for i, result in enumerate(results["individual_results"][:10]):  # Show first 10
                if "error" not in result:
                    f.write(f"### Question {result.get('question_id', f'q{i+1}')}: {result.get('question', 'N/A')}\n\n")
                    f.write(f"**Answer:** {result.get('generation', {}).get('answer', 'N/A')[:200]}...\n\n")
                    f.write("**Metrics:**\n")
                    retrieval = result.get("retrieval", {})
                    generation = result.get("generation", {})
                    for metric, value in retrieval.items():
                        f.write(f"- {metric}: {value:.3f}\n")
                    for metric, value in generation.items():
                        if metric != "answer":  # Skip the actual answer text
                            f.write(f"- {metric}: {value:.3f}\n")
                    f.write("\n---\n\n")
                else:
                    f.write(f"### Question {result.get('question_id', f'q{i+1}')}: ERROR\n\n")
                    f.write(f"**Error:** {result.get('error', 'Unknown error')}\n\n")
                    f.write("---\n\n")


def run_comparison_evaluation(golden_set_path: str, output_dir: str) -> None:
    """Run several pipeline configurations and write a comparison report.

    Args:
        golden_set_path: Path to golden set JSON file
        output_dir: Directory to save results

    WHAT: Defines six configurations (baseline through advanced_pipeline),
    evaluates each on the golden set, saves per-config JSON and a combined
    markdown comparison table.
    WHY: Lets stakeholders compare configurations side by side on the same
    golden set.
    INPUTS: golden_set_path string, output_dir string.
    OUTPUTS: none (writes files to output_dir).
    SIDE EFFECTS: creates output_dir, writes one JSON per config and a
        comparison_results.md; re-ingests documents per config.
    FAILURE MODES: a failing config aborts the whole comparison (no per-config
        try/except).
    ALTERNATIVE: run configurations in parallel to cut wall-clock time.
    # GAP: concurrency - configurations run sequentially; each re-ingests the
        corpus, so wall-clock is linear in the number of configs.
    # BUSINESS: the side-by-side report used to pick a production configuration.
    # INTERVIEW: "Why not parallelize the configs?" - each config re-ingests and
        re-indexes; parallelizing would multiply concurrent disk/vector-store
        load and complicate result aggregation.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define configurations to compare
    configs = [
        {
            "name": "baseline",
            "chunk_strategy": "fixed_size",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "use_hybrid": False,
            "use_reranker": False,
            "use_rrf": False,
            "hybrid_alpha": 0.7,
            "prompt_version": "grounded_v1"
        },
        {
            "name": "advanced_chunking",
            "chunk_strategy": "recursive",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "use_hybrid": False,
            "use_reranker": False,
            "use_rrf": False,
            "hybrid_alpha": 0.7,
            "prompt_version": "grounded_v1"
        },
        {
            "name": "hybrid_retrieval",
            "chunk_strategy": "recursive",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "use_hybrid": True,
            "use_reranker": False,
            "use_rrf": False,
            "hybrid_alpha": 0.7,
            "prompt_version": "grounded_v1"
        },
        {
            "name": "hybrid_rrf",
            "chunk_strategy": "recursive",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "use_hybrid": True,
            "use_reranker": False,
            "use_rrf": True,
            "hybrid_alpha": 0.7,
            "prompt_version": "grounded_v1"
        },
        {
            "name": "full_pipeline",
            "chunk_strategy": "recursive",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "use_hybrid": True,
            "use_reranker": True,
            "use_rrf": False,
            "hybrid_alpha": 0.7,
            "prompt_version": "grounded_v1"
        },
        {
            "name": "advanced_pipeline",
            "chunk_strategy": "recursive",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "use_hybrid": True,
            "use_reranker": True,
            "use_rrf": True,
            "hybrid_alpha": 0.7,
            "prompt_version": "grounded_v1"
        }
    ]

    # Run evaluation for each configuration
    all_results = {}

    for config in configs:
        logger.info(f"Running evaluation for configuration: {config['name']}")

        # Create pipeline
        pipeline = RAGEvaluationPipeline(
            chunk_strategy=config["chunk_strategy"],
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            use_hybrid=config["use_hybrid"],
            use_reranker=config["use_reranker"],
            use_rrf=config["use_rrf"],
            hybrid_alpha=config["hybrid_alpha"],
            prompt_version=config["prompt_version"]
        )

        # Ingest and index documents
        data_directory = project_root / "data" / "raw"
        pipeline.ingest_and_index_documents(str(data_directory))

        # Run evaluation
        results = pipeline.evaluate_golden_set(golden_set_path)
        all_results[config["name"]] = results

        # Save individual results
        output_file = os.path.join(output_dir, f"{config['name']}_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved results for {config['name']} to {output_file}")

    # Create comparison markdown report
    comparison_file = os.path.join(output_dir, "comparison_results.md")
    with open(comparison_file, 'w') as f:
        f.write("# RAG System Configuration Comparison\n\n")

        f.write("## Configuration Overview\n\n")
        f.write("| Configuration | Chunking | Hybrid | RRF | Reranker | Prompt Version |\n")
        f.write("|---------------|----------|--------|-----|----------|----------------|\n")
        for config in configs:
            name = config["name"]
            chunking = config["chunk_strategy"]
            hybrid = "✓" if config["use_hybrid"] else "✗"
            rrf = "✓" if config["use_rrf"] else "✗"
            reranker = "✓" if config["use_reranker"] else "✗"
            prompt = config["prompt_version"]
            f.write(f"| {name} | {chunking} | {hybrid} | {rrf} | {reranker} | {prompt} |\n")
        f.write("\n")

        f.write("## Performance Comparison\n\n")
        f.write("| Configuration | Hit Rate@5 | Recall@5 | MRR | Answer Relevance | Faithfulness |\n")
        f.write("|---------------|------------|----------|-----|------------------|--------------|\n")
        for config in configs:
            name = config["name"]
            if name in all_results and "aggregate_metrics" in all_results[name]:
                agg = all_results[name]["aggregate_metrics"]
                hit_rate_5 = agg.get("retrieval", {}).get("hit_rate_at_5", 0.0)
                recall_5 = agg.get("retrieval", {}).get("recall_at_5", 0.0)
                mrr = agg.get("retrieval", {}).get("mrr", 0.0)
                answer_rel = agg.get("generation", {}).get("answer_relevance", 0.0)
                faithful = agg.get("generation", {}).get("faithfulness", 0.0)
                f.write(f"| {name} | {hit_rate_5:.3f} | {recall_5:.3f} | {mrr:.3f} | {answer_rel:.3f} | {faithful:.3f} |\n")
            else:
                f.write(f"| {name} | N/A | N/A | N/A | N/A | N/A |\n")
        f.write("\n")

        f.write("## Key Findings\n\n")
        f.write("1. **Impact of Chunking Strategy**: Comparing baseline vs advanced_chunking shows the effect of better chunking.\n")
        f.write("2. **Impact of Hybrid Retrieval**: Comparing advanced_chunking vs hybrid_retrieval shows the value of BM25+dense fusion.\n")
        f.write("3. **Impact of RRF**: Comparing hybrid_retrieval vs hybrid_rrf shows the value of Reciprocal Rank Fusion.\n")
        f.write("4. **Impact of Reranking**: Comparing hybrid_rrf vs full_pipeline shows the value of cross-encoder reranking.\n")
        f.write("5. **Impact of Advanced Prompting**: Comparing full_pipeline vs advanced_pipeline shows the value of improved prompts.\n")


def main():
    """CLI entry point: run standard, ablation, or comparison evaluation.

    WHAT: Parses argparse flags, builds the pipeline (with or without LLM judge),
    ingests documents, and runs either an ablation study, a configuration
    comparison, or a standard single evaluation, writing results to disk.
    WHY: Provides the command-line interface developers/CI use to measure RAG
    quality before deploy.
    INPUTS: CLI args (--data-dir, --golden-set, --output, --chunk-*,
        --use-hybrid/--no-hybrid, --use-reranker/--no-reranker,
        --use-rrf/--no-rrf, --hybrid-alpha, --prompt-version,
        --run-ablation, --run-comparison, --output-dir, --no-judge).
    OUTPUTS: none (writes markdown/JSON files; logs progress).
    SIDE EFFECTS: creates output directories, writes files, ingests/indexes
        documents, may invoke the LLM judge (network/cost).
    FAILURE MODES: missing golden-set file raises FileNotFoundError; unwritable
        output raises IOError; judge errors degrade to "0" scores (see
        make_llm_judge).
    ALTERNATIVE: call RAGEvaluationPipeline directly from Python instead of the
        CLI.
    # GAP: resilience - no retry on ingestion or judge failure; a transient
        error aborts the run.
    # BUSINESS: is the entry point CI/developers use to gate telecom QA quality
        changes.
    # INTERVIEW: "Why default --use-hybrid and --use-reranker to True?" -
        because the recommended production configuration enables both; the
        flags let you opt out for ablation.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing source documents")
    parser.add_argument("--golden-set", default="src/evaluation/golden_set.json", help="Path to golden set JSON file")
    parser.add_argument("--output", default="evaluation/results.md", help="Output file for results")
    parser.add_argument("--chunk-strategy", default="recursive", help="Chunking strategy (fixed_size, recursive, structure_aware)")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap")
    parser.add_argument("--use-hybrid", action="store_true", help="Use hybrid retrieval")
    parser.add_argument("--no-hybrid", dest="use_hybrid", action="store_false", help="Use dense retrieval only")
    parser.add_argument("--use-reranker", action="store_true", help="Use reranker")
    parser.add_argument("--no-reranker", dest="use_reranker", action="store_false", help="Do not use reranker")
    parser.add_argument("--use-rrf", action="store_true", help="Use RRF for hybrid retrieval")
    parser.add_argument("--no-rrf", dest="use_rrf", action="store_false", help="Do not use RRF for hybrid retrieval")
    parser.add_argument("--hybrid-alpha", type=float, default=0.7, help="Weight for dense retrieval in hybrid (0-1)")
    parser.add_argument("--prompt-version", default="grounded_v1", help="Prompt version to use")
    parser.add_argument("--run-ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--run-comparison", action="store_true", help="Run configuration comparison")
    parser.add_argument("--output-dir", default="evaluation", help="Output directory for multiple runs")
    parser.add_argument("--no-judge", action="store_true",
                        help="Disable LLM-as-judge (heuristic fallbacks only)")

    # Set defaults
    parser.set_defaults(use_hybrid=True, use_reranker=True, use_rrf=False)

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    if args.run_ablation:
        logger.info("Running ablation study...")
        # Initialize pipeline with base config
        pipeline = RAGEvaluationPipeline(
            chunk_strategy=args.chunk_strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            use_hybrid=args.use_hybrid,
            use_reranker=args.use_reranker,
            use_rrf=args.use_rrf,
            hybrid_alpha=args.hybrid_alpha,
            prompt_version=args.prompt_version
        )

        # Ingest and index documents
        data_directory = project_root / args.data_dir
        pipeline.ingest_and_index_documents(str(data_directory))

        # Run ablation study
        golden_set_path = project_root / args.golden_set
        ablation_results = pipeline.run_ablation_study(str(golden_set_path), {
            "chunk_strategy": args.chunk_strategy,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "use_hybrid": args.use_hybrid,
            "use_reranker": args.use_reranker,
            "use_rrf": args.use_rrf,
            "hybrid_alpha": args.hybrid_alpha,
            "prompt_version": args.prompt_version
        })

        # Save ablation results
        ablation_output = os.path.join(args.output_dir, "ablation_results.json")
        with open(ablation_output, 'w') as f:
            json.dump(ablation_results, f, indent=2)

        logger.info(f"Ablation study complete. Results saved to {ablation_output}")

        # Also save summary to markdown
        ablation_md = os.path.join(args.output_dir, "ablation_summary.md")
        with open(ablation_md, 'w') as f:
            f.write("# Ablation Study Results\n\n")
            f.write("This study shows the impact of each component on overall performance.\n\n")
            f.write("## Configurations Tested\n\n")
            f.write("| Component | Description |\n")
            f.write("|-----------|-------------|\n")
            f.write("| Baseline | Fixed size chunking, dense retrieval only, basic prompt |\n")
            f.write("| Chunking Improvement | Recursive chunking, dense retrieval only, basic prompt |\n")
            f.write("| Hybrid Retrieval | Recursive chunking, BM25+dense retrieval, basic prompt |\n")
            f.write("| Hybrid + RRF | Recursive chunking, BM25+dense+RRF retrieval, basic prompt |\n")
            f.write("| Reranking | Recursive chunking, BM25+dense+RRF retrieval + reranking, basic prompt |\n")
            f.write("| Advanced Prompting | Recursive chunking, BM25+dense+RRF retrieval + reranking + advanced prompt |\n\n")

            f.write("## Key Metrics Comparison\n\n")
            f.write("| Configuration | Hit Rate@5 | Recall@5 | MRR | Answer Relevance | Faithfulness |\n")
            f.write("|---------------|------------|----------|-----|------------------|--------------|\n")
            configs_order = ["baseline", "chunking_improvement", "hybrid_retrieval", "hybrid_with_rrf", "reranking", "advanced_prompting"]
            for config_name in configs_order:
                if config_name in ablation_results and "aggregate_metrics" in ablation_results[config_name]:
                    agg = ablation_results[config_name]["aggregate_metrics"]
                    hit_rate_5 = agg.get("retrieval", {}).get("hit_rate_at_5", 0.0)
                    recall_5 = agg.get("retrieval", {}).get("recall_at_5", 0.0)
                    mrr = agg.get("retrieval", {}).get("mrr", 0.0)
                    answer_rel = agg.get("generation", {}).get("answer_relevance", 0.0)
                    faithful = agg.get("generation", {}).get("faithfulness", 0.0)
                    f.write(f"| {config_name} | {hit_rate_5:.3f} | {recall_5:.3f} | {mrr:.3f} | {answer_rel:.3f} | {faithful:.3f} |\n")
                else:
                    f.write(f"| {config_name} | N/A | N/A | N/A | N/A | N/A |\n")

    elif args.run_comparison:
        logger.info("Running configuration comparison...")
        golden_set_path = project_root / args.golden_set
        run_comparison_evaluation(str(golden_set_path), args.output_dir)
        logger.info(f"Configuration comparison complete. Results saved to {args.output_dir}")

    else:
        # Standard single evaluation
        logger.info("Running standard evaluation...")

        # Initialize pipeline (real judge from env unless --no-judge)
        judge_fn = None if args.no_judge else make_llm_judge()
        pipeline = RAGEvaluationPipeline(
            chunk_strategy=args.chunk_strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            use_hybrid=args.use_hybrid,
            use_reranker=args.use_reranker,
            use_rrf=args.use_rrf,
            hybrid_alpha=args.hybrid_alpha,
            prompt_version=args.prompt_version,
            llm_judge_fn=judge_fn
        )

        # Ingest and index documents
        data_directory = project_root / args.data_dir
        pipeline.ingest_and_index_documents(str(data_directory))

        # Run evaluation
        golden_set_path = project_root / args.golden_set
        results = pipeline.evaluate_golden_set(str(golden_set_path))

        # Save results as markdown
        save_results_to_markdown(results, args.output)

        logger.info(f"Evaluation complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()