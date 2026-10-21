"""
Query rewriting utilities for improving retrieval quality.
Includes LLM-based rewrite, multi-query expansion, and HyDE (Hypothetical Document Embedding).
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import re

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Handles various query rewriting techniques to improve retrieval."""

    def __init__(self, llm_provider: Any = None):
        """
        Initialize the query rewriter.

        Args:
            llm_provider: LLM provider for query rewriting (optional)
        """
        self.llm_provider = llm_provider
        logger.info(f"Initialized QueryRewriter with LLM provider: {llm_provider is not None}")

    def rewrite_query(self, query: str, technique: str = "none") -> str:
        """
        Rewrite a query using the specified technique.

        Args:
            query: Original query string
            technique: Rewriting technique ("none", "llm", "hyde")

        Returns:
            Rewritten query string
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
        Expand a query into multiple variations.

        Args:
            query: Original query string
            technique: Expansion technique ("none", "multi-query")
            num_queries: Number of queries to generate

        Returns:
            List of query strings (including original)
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
        Rewrite query using LLM to make it more specific and retrieval-friendly.

        Args:
            query: Original query string

        Returns:
            Rewritten query string
        """
        try:
            # Simple rule-based rewriting as fallback when LLM not available
            # In a real implementation, this would call the LLM provider
            rewritten = self._apply_rewriting_rules(query)
            logger.info(f"Rewrote query: '{query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.error(f"Error in LLM-based query rewrite: {e}")
            return query

    def _hyde_rewrite(self, query: str) -> str:
        """
        Generate a hypothetical document embedding (HyDE) - creates a pseudo-document
        that answers the query, then uses that for embedding.

        Args:
            query: Original query string

        Returns:
            Hypothetical document string
        """
        try:
            # Simple hypothetical document generation
            # In a real implementation, this would use an LLM to generate
            # a document that answers the query
            hypothetical = self._generate_hypothetical_document(query)
            logger.info(f"Generated HyDE document for query: '{query}'")
            return hypothetical
        except Exception as e:
            logger.error(f"Error in HyDE rewrite: {e}")
            return query

    def _multi_query_expansion(self, query: str, num_queries: int) -> List[str]:
        """
        Generate multiple query variations using LLM.

        Args:
            query: Original query string
            num_queries: Number of queries to generate

        Returns:
            List of query strings
        """
        try:
            # Generate variations using simple techniques as fallback
            queries = [query]  # Start with original

            # Add some simple variations
            variations = self._generate_query_variations(query)
            queries.extend(variations[:num_queries-1])  # Add variations up to num_queries-1

            logger.info(f"Expanded query into {len(queries)} variations")
            return queries[:num_queries]  # Ensure we don't exceed requested count
        except Exception as e:
            logger.error(f"Error in multi-query expansion: {e}")
            return [query]

    def _apply_rewriting_rules(self, query: str) -> str:
        """
        Apply simple rule-based rewriting as fallback.

        Args:
            query: Original query string

        Returns:
            Rewritten query string
        """
        # Convert to lowercase for processing
        lower_query = query.lower().strip()

        # Expand common abbreviations in telecom domain
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

        # Apply abbreviation expansions
        expanded = lower_query
        for abbr, full_form in abbreviations.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbr) + r'\b'
            expanded = re.sub(pattern, full_form, expanded, flags=re.IGNORECASE)

        # Add context for vague pronouns
        if "it" in expanded.split() and len(expanded.split()) <= 3:
            # Very short queries with "it" might benefit from context
            # This is a simple heuristic - in practice, would need conversation history
            pass

        # Return with original casing preserved but expanded
        # For simplicity, we return the expanded version
        # In a real system, we might preserve more of the original formatting
        return expanded.strip()

    def _generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document that answers the query (HyDE).

        Args:
            query: Original query string

        Returns:
            Hypothetical document string
        """
        # Simple template-based hypothetical document generation
        # In a real implementation, this would use an LLM

        query_lower = query.lower().strip()

        # Detect question type and generate appropriate hypothetical answer
        if any(word in query_lower for word in ["what", "who", "when", "where"]):
            # Factual question
            hypothetical = f"The answer to {query} is based on established telecom standards and specifications. This information can be found in the relevant technical documentation covering network architecture, protocols, and procedures."
        elif any(word in query_lower for word in ["how", "why"]):
            # Explanatory question
            hypothetical = f"The process described in {query} involves multiple steps following standardized telecommunications procedures. Implementation requires compliance with relevant specifications and includes verification steps to ensure proper operation."
        elif any(word in query_lower for word in ["compare", "difference", "versus", "vs"]):
            # Comparison question
            hypothetical = f"When comparing the subjects in {query}, several key differences emerge based on technical specifications, implementation requirements, and operational characteristics. Each approach has distinct advantages depending on the specific use case and deployment scenario."
        else:
            # General query
            hypothetical = f"Information regarding {query} can be found in telecommunications documentation covering relevant standards, procedures, and best practices. Details include technical specifications, implementation guidelines, and operational considerations."

        return hypothetical

    def _generate_query_variations(self, query: str) -> List[str]:
        """
        Generate simple query variations.

        Args:
            query: Original query string

        Returns:
            List of query variations
        """
        variations = []
        query_lower = query.lower().strip()

# Variation 1: Add context words
        if len(query.split()) <= 4:  # Short query
            variations.append(f"information about {query}")
            variations.append(f"details regarding {query}")
            variations.append(f"explanation of {query}")

# Variation 2: Question format conversion
        if not query.endswith('?'):
            if any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where']):
                variations.append(f"{query}?")
            else:
                variations.append(f"What is {query}?")

# Variation 3: Add domain context
        telecom_terms = ['network', 'protocol', 'specification', 'standard', 'procedure']
        if not any(term in query_lower for term in telecom_terms):
            variations.append(f"{query} in telecommunications")
            variations.append(f"telecom {query}")

        # Limit variations and remove duplicates
        unique_variations = list(dict.fromkeys(variations))  # Preserves order
        return unique_variations[:5]  # Return max 5 variations


# Convenience functions
def create_query_rewriter(llm_provider: Any = None) -> QueryRewriter:
    """Create a query rewriter instance."""
    return QueryRewriter(llm_provider)


def rewrite_query(query: str, technique: str = "none", llm_provider: Any = None) -> str:
    """Convenience function to rewrite a query."""
    rewriter = QueryRewriter(llm_provider)
    return rewriter.rewrite_query(query, technique)


def expand_query(query: str, technique: str = "none", num_queries: int = 3, llm_provider: Any = None) -> List[str]:
    """Convenience function to expand a query."""
    rewriter = QueryRewriter(llm_provider)
    return rewriter.expand_query(query, technique, num_queries)