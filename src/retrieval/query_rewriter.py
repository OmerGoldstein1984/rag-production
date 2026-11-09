"""
Query rewriting utilities for improving retrieval quality.
Includes LLM-based rewrite, multi-query expansion, and HyDE (Hypothetical Document Embedding).

FILE HEADER
===========
Business problem:
  User questions are often short, vague, or full of telecom acronyms (e.g. "TAC?",
  "eSIM how?"). Retrieval works best with a clear, self-contained query, so this
  module rewrites or expands a question into retrieval-friendly forms before it
  reaches the vector store.

Place in data flow:
  Sits in G4 Retrieval, before retrieval (dense/sparse/hybrid). Input: the user's
  raw question. Output: a rewritten query string, or a list of expanded queries.

What calls it:
  The chat / agent layer (G5) calls QueryRewriter.rewrite_query() or
  expand_query() to turn a follow-up or short question into something retrievable.

What it calls:
  llm_provider (optional, injected) — in this version the LLM path is a placeholder
  and rule-based / template-based methods are used as fallback; re and logging only
  otherwise. No external service is actually called.

What breaks if removed:
  Query rewriting disappears; raw short/ambiguous questions go straight to
  retrieval, which often returns weak results for acronym-heavy or pronoun-heavy
  telecom questions.

Tags:
  # GAP: LLM integration — the LLM provider is accepted but never actually invoked;
         _llm_based_rewrite, _hyde_rewrite, and _multi_query_expansion fall back to
         rule-based logic, so real LLM-powered rewriting is unimplemented.
  # BUSINESS: protects recall for underspecified questions — expanding acronyms and
              generating variations helps retrieval find the right telecom chunks.
  # INTERVIEW: say-it-out-loud — "HyDE invents a hypothetical document answering the
               question, embeds that, and retrieves by similarity to the answer
               rather than to the short query — helpful when the real answer uses
               different words than the question."
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import re

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Handles various query rewriting techniques to improve retrieval."""

    def __init__(self, llm_provider: Any = None):
        """
        WHAT:  Store an optional LLM provider used for rewriting.
        WHY:   Rewriting can be done by rules (always available) or by an LLM
               (better quality, optional) — injection lets tests use fakes and
               production use a real provider.
        INPUTS: llm_provider (Any): object capable of generating text; None means
                rule-based fallback only.
        OUTPUTS: None (instance initialized).
        SIDE EFFECTS: Logs an info line with whether a provider was supplied
                      (query_rewriter.py:24).
        FAILURE MODES: None at construction.
        ALTERNATIVE: Always use rule-based rewriting when you cannot depend on an
                     LLM provider being available or configured.
        """
        self.llm_provider = llm_provider
        logger.info(f"Initialized QueryRewriter with LLM provider: {llm_provider is not None}")

    def rewrite_query(self, query: str, technique: str = "none") -> str:
        """
        WHAT:  Rewrite a single query using the named technique.
        WHY:   A clearer query improves retrieval recall; "none" returns the query
               unchanged when no rewriting is desired.
        INPUTS:
            query (str): e.g. "TAC handover?"
            technique (str): "none", "llm", or "hyde".
        OUTPUTS: str — the rewritten query (or the original if technique unknown /
                 provider unavailable).
        SIDE EFFECTS: Logs a warning for unknown techniques (query_rewriter.py:44).
        FAILURE MODES: Unknown technique → returns the original query (no error).
                       LLM-backed techniques degrade to rule-based output if the
                       provider path fails internally.
        ALTERNATIVE: Use expand_query() when you want multiple candidate queries
                     rather than a single rewrite.
        """
        if technique == "none" or self.llm_provider is None:
            return query
        elif technique == "llm":
            return self._llm_based_rewrite(query)
        elif technique == "hyde":
            return self._hyde_rewrite(query)
        else:
            logger.warning(f"Unknown query rewriting technique: {technique}")
            return query

    def expand_query(self, query: str, technique: str = "none", num_queries: int = 3) -> List[str]:
        """
        WHAT:  Expand a query into multiple variations for retrieval.
        WHY:   Retrieving with several phrasings increases the chance that at least
               one matches the stored chunks, especially for short user questions.
        INPUTS:
            query (str): e.g. "eSIM provisioning"
            technique (str): "none" or "multi-query".
            num_queries (int): 3 — maximum variations to return.
        OUTPUTS: List[str] including the original query — e.g.
                 ["eSIM provisioning", "information about eSIM provisioning", ...]
        SIDE EFFECTS: Logs a warning for unknown techniques (query_rewriter.py:64).
        FAILURE MODES: Unknown technique or no provider → returns [query] only.
                       Internal errors return [query] as a safe fallback.
        ALTERNATIVE: Use rewrite_query() for a single rewritten query instead of
                     a list of variations.
        """
        if technique == "none" or self.llm_provider is None:
            return [query]
        elif technique == "multi-query":
            return self._multi_query_expansion(query, num_queries)
        else:
            logger.warning(f"Unknown query expansion technique: {technique}")
            return [query]

    def _llm_based_rewrite(self, query: str) -> str:
        """
        WHAT:  Rewrite the query using rule-based rules as a stand-in for an LLM.
        WHY:   The real LLM call is not implemented; the rule-based expansion
               (abbreviation expansion) is the concrete, testable behaviour today.
        INPUTS: query (str): the original query string.
        OUTPUTS: str — the rewritten (abbreviation-expanded) query.
        SIDE EFFECTS: Logs an info line showing the rewrite (query_rewriter.py:81).
        FAILURE MODES: Exceptions are caught, logged, and the original query is
                       returned (no exception propagates).
        ALTERNATIVE: Replace _apply_rewriting_rules with an actual LLM call to
                     self.llm_provider when a provider is configured.
        """
        try:
            # Simple rule-based rewriting as fallback when LLM not available.
            # In a real implementation, this would call the LLM provider.
            rewritten = self._apply_rewriting_rules(query)
            logger.info(f"Rewrote query: '{query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.error(f"Error in LLM-based query rewrite: {e}")
            return query

    def _hyde_rewrite(self, query: str) -> str:
        """
        WHAT:  Generate a hypothetical document that answers the query (HyDE).
        WHY:   Embedding a plausible answer often retrieves better than embedding
               the short question itself, because real answers use different words.
        INPUTS: query (str): the original query string.
        OUTPUTS: str — a hypothetical document string (template-based here).
        SIDE EFFECTS: Logs an info line (query_rewriter.py:103).
        FAILURE MODES: Exceptions caught, logged, and the original query returned.
                       The HyDE text is a fixed template, not a real generated answer,
                       so quality is limited.
        ALTERNATIVE: Use an actual LLM to generate the hypothetical document when
                     self.llm_provider is available.
        """
        try:
            # Simple hypothetical document generation.
            # In a real implementation, this would use an LLM to generate
            # a document that answers the query.
            hypothetical = self._generate_hypothetical_document(query)
            logger.info(f"Generated HyDE document for query: '{query}'")
            return hypothetical
        except Exception as e:
            logger.error(f"Error in HyDE rewrite: {e}")
            return query

    def _multi_query_expansion(self, query: str, num_queries: int) -> List[str]:
        """
        WHAT:  Generate multiple query variations from the original.
        WHY:   More phrasings = more retrieval shots on target; variations include
               the original plus rule-generated alternatives.
        INPUTS: query (str): the original query; num_queries (int): max count.
        OUTPUTS: List[str] — up to num_queries variations including the original.
        SIDE EFFECTS: Logs an info line with the count (query_rewriter.py:128).
        FAILURE MODES: Exceptions caught, logged, and [query] returned as fallback.
        ALTERNATIVE: Use an LLM to paraphrase the question into diverse variations
                     instead of the simple templates here.
        """
        try:
            # Generate variations using simple techniques as fallback.
            queries = [query]  # Start with original

            # Add some simple variations.
            variations = self._generate_query_variations(query)
            queries.extend(variations[:num_queries-1])  # Add variations up to num_queries-1

            logger.info(f"Expanded query into {len(queries)} variations")
            return queries[:num_queries]  # Ensure we don't exceed requested count
        except Exception as e:
            logger.error(f"Error in multi-query expansion: {e}")
            return [query]

    def _apply_rewriting_rules(self, query: str) -> str:
        """
        WHAT:  Expand telecom abbreviations to their full forms in the query.
        WHY:   Telecom questions use acronyms (LTE, IMS, QoS, eSIM) that may not
               appear verbatim in stored chunks; expanding them improves lexical
               and semantic matching.
        INPUTS: query (str): e.g. "What is the TAC?"
        OUTPUTS: str — the query with abbreviations lowercased and expanded.
        SIDE EFFECTS: None (pure string transformation).
        FAILURE MODES: Never raises; abbreviations absent from the map are left
                       unchanged (lowercased).
        ALTERNATIVE: Maintain the abbreviation map in a config file instead of
                     hard-coding it, so domain experts can edit it without code.
        """
        # Convert to lowercase for processing.
        lower_query = query.lower().strip()

        # Expand common abbreviations in telecom domain.
        abbreviations = {
            "lte": "Long Term Evolution",
            "5g": "5th Generation",
            "4g": "4th Generation",
            "3g": "3rd Generation",
            "vo lte": "Voice over LTE",
            " VoLTE": "Voice over LTE",
            "ims": "IP Multimedia Subsystem",
            "qos": "Quality of Service",
            "sim": "Subscriber Identity Module",
            "esim": "Embedded SIM",
            "imei": "International Mobile Equipment Identity",
            "msisdn": "Mobile Station International Subscriber Directory Number",
            "iccid": "Integrated Circuit Card Identifier",
            "plmn": "Public Land Mobile Network",
            "tac": "Tracking Area Code",
            "cell id": "Cell Identifier",
            "ecgi": "E-UTRAN Cell Global Identifier",
            "ncgi": "NR Cell Global Identifier",
            "ran": "Radio Access Network",
            "cn": "Core Network",
            "ue": "User Equipment",
            "enb": "Evolved Node B",
            "gnb": "gNodeB",
            "ngran": "Next Generation Radio Access Network"
        }

        # Apply abbreviation expansions.
        expanded = lower_query
        for abbr, full_form in abbreviations.items():
            # Use word boundaries to avoid partial matches.
            pattern = r'\b' + re.escape(abbr) + r'\b'
            expanded = re.sub(pattern, full_form, expanded, flags=re.IGNORECASE)

        # Add context for vague pronouns (short queries with "it").
        if "it" in expanded.split() and len(expanded.split()) <= 3:
            # Very short queries with "it" might benefit from context.
            # This is a simple heuristic - in practice, would need conversation history.
            pass

        # Return the expanded version with original casing not preserved.
        # For simplicity we return the expanded version.
        return expanded.strip()

    def _generate_hypothetical_document(self, query: str) -> str:
        """
        WHAT:  Build a template-based hypothetical document answering the query.
        WHY:   HyDE retrieves by similarity to a plausible answer; the template
               picks a form (factual / explanatory / comparison / general) based on
               the question type.
        INPUTS: query (str): the original query string.
        OUTPUTS: str — a hypothetical document paragraph.
        SIDE EFFECTS: None (pure string construction).
        FAILURE MODES: Never falls through to an error; the else branch always
                       returns a generic document.
        ALTERNATIVE: Generate the hypothetical document with an LLM for richer,
                     question-specific text instead of these fixed templates.
        """
        # Simple template-based hypothetical document generation.
        # In a real implementation, this would use an LLM.

        query_lower = query.lower().strip()

        # Detect question type and generate appropriate hypothetical answer.
        if any(word in query_lower for word in ["what", "who", "when", "where"]):
            # Factual question.
            hypothetical = f"The answer to {query} is based on established telecom standards and specifications. This information can be found in the relevant technical documentation covering network architecture, protocols, and procedures."
        elif any(word in query_lower for word in ["how", "why"]):
            # Explanatory question.
            hypothetical = f"The process described in {query} involves multiple steps following standardized telecommunications procedures. Implementation requires compliance with relevant specifications and includes verification steps to ensure proper operation."
        elif any(word in query_lower for word in ["compare", "difference", "versus", "vs"]):
            # Comparison question.
            hypothetical = f"When comparing the subjects in {query}, several key differences emerge based on technical specifications, implementation requirements, and operational characteristics. Each approach has distinct advantages depending on the specific use case and deployment scenario."
        else:
            # General query.
            hypothetical = f"Information regarding {query} can be found in telecommunications documentation covering relevant standards, procedures, and best practices. Details include technical specifications, implementation guidelines, and operational considerations."

        return hypothetical

    def _generate_query_variations(self, query: str) -> List[str]:
        """
        WHAT:  Produce simple rule-based query variations.
        WHY:   Different phrasings (adding context words, converting to questions,
               adding domain terms) increase retrieval coverage for short queries.
        INPUTS: query (str): the original query string.
        OUTPUTS: List[str] — up to 5 unique variations.
        SIDE EFFECTS: None (pure string construction).
        FAILURE MODES: Returns an empty list when no variation rule matches — the
                       caller (_multi_query_expansion) still includes the original.
        ALTERNATIVE: Use an LLM to paraphrase for more natural, diverse variations.
        """
        variations = []
        query_lower = query.lower().strip()

        # Variation 1: Add context words (for short queries).
        if len(query.split()) <= 4:  # Short query
            variations.append(f"information about {query}")
            variations.append(f"details regarding {query}")
            variations.append(f"explanation of {query}")

        # Variation 2: Question format conversion.
        if not query.endswith('?'):
            if any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where']):
                variations.append(f"{query}?")
            else:
                variations.append(f"What is {query}?")

        # Variation 3: Add domain context.
        telecom_terms = ['network', 'protocol', 'specification', 'standard', 'procedure']
        if not any(term in query_lower for term in telecom_terms):
            variations.append(f"{query} in telecommunications")
            variations.append(f"telecom {query}")

        # Limit variations and remove duplicates (dict.fromkeys preserves order).
        unique_variations = list(dict.fromkeys(variations))
        return unique_variations[:5]  # Return max 5 variations


# Convenience functions
def create_query_rewriter(llm_provider: Any = None) -> QueryRewriter:
    """
    WHAT:  Build and return a QueryRewriter instance.
    WHY:   Factory convenience for the query rewriter.
    INPUTS: llm_provider (Any): optional LLM provider.
    OUTPUTS: QueryRewriter.
    SIDE EFFECTS: None.
    FAILURE MODES: None at construction.
    ALTERNATIVE: Instantiate QueryRewriter(llm_provider) directly.
    """
    return QueryRewriter(llm_provider)


def rewrite_query(query: str, technique: str = "none", llm_provider: Any = None) -> str:
    """
    WHAT:  One-shot convenience to rewrite a query.
    WHY:   Lets callers avoid constructing a QueryRewriter for a single rewrite.
    INPUTS: query (str), technique (str): "none", llm_provider (Any).
    OUTPUTS: str — the rewritten query.
    SIDE EFFECTS: Constructs and discards a QueryRewriter (minor overhead).
    FAILURE MODES: Delegates to QueryRewriter.rewrite_query(); same behaviour.
    ALTERNATIVE: Reuse a single QueryRewriter instance across many calls.
    """
    rewriter = QueryRewriter(llm_provider)
    return rewriter.rewrite_query(query, technique)


def expand_query(query: str, technique: str = "none", num_queries: int = 3, llm_provider: Any = None) -> List[str]:
    """
    WHAT:  One-shot convenience to expand a query into variations.
    WHY:   Lets callers avoid constructing a QueryRewriter for a single expansion.
    INPUTS: query (str), technique (str), num_queries (int), llm_provider (Any).
    OUTPUTS: List[str] — query variations.
    SIDE EFFECTS: Constructs and discards a QueryRewriter (minor overhead).
    FAILURE MODES: Delegates to QueryRewriter.expand_query(); same behaviour.
    ALTERNATIVE: Reuse a single QueryRewriter instance across many calls.
    """
    rewriter = QueryRewriter(llm_provider)
    return rewriter.expand_query(query, technique, num_queries)
