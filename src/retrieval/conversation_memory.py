"""
Conversation memory utilities for condensing follow-up questions into standalone queries.

FILE HEADER
===========
Business problem:
  In a chat, users ask follow-ups like "What about it?" or "How does that work?"
  that only make sense with previous context. This module keeps a short history
  and condenses a follow-up into a standalone query so retrieval can run without
  losing the conversation thread.

Place in data flow:
  Sits in G4 Retrieval, at the start of a chat turn — before query rewriting and
  retrieval. Input: current question + prior history. Output: a standalone query
  string ready for the retriever.

What calls it:
  The chat / agent layer (G5) calls condense_question() or
  condense_follow_up_question() to turn a follow-up into a retrievable query.

What it calls:
  datetime / timedelta for timestamps; re for pattern matching; llm_provider
  (optional, injected) — not actually invoked in this version (placeholder).

What breaks if removed:
  Follow-up questions lose their context and are sent to retrieval as-is, so
  pronouns ("it", "that") and references ("the previous one") cannot be resolved
  and retrieval returns weak or unrelated chunks.

Tags:
  # GAP: persistence / concurrency — history is an in-memory list (self.history)
         with no persistence, no TTL eviction beyond max_history_length, and no
         locking for concurrent turns. Also the LLM condensation path is a
         placeholder that always falls back to rules.
  # BUSINESS: protects multi-turn answer correctness — without context resolution,
              follow-ups retrievable poorly and answers drift off-topic.
  # INTERVIEW: say-it-out-loud — "We condense 'what about it?' into a standalone
               question by pulling entities from the last turn, so the retriever
               sees a complete query instead of a dangling pronoun."
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Manages conversation history and condenses follow-up questions into standalone queries."""

    def __init__(self, max_history_length: int = 10, llm_provider: Any = None):
        """
        WHAT:  Configure history length, store an optional LLM provider, and
               initialize an empty history list.
        WHY:   Bounding history length prevents unbounded memory growth in long
               chats; the LLM provider is optional so the class works without one.
        INPUTS: max_history_length (int): 10 — most recent turns to keep.
                llm_provider (Any): optional provider for LLM-based condensation.
        OUTPUTS: None (instance initialized with empty history).
        SIDE EFFECTS: Logs an info line with config (conversation_memory.py:28).
        FAILURE MODES: None at construction.
        ALTERNATIVE: Use a persisted chat store (database) when history must
                     survive restarts or be shared across servers.
        """
        self.max_history_length = max_history_length
        self.llm_provider = llm_provider
        self.history: List[Dict[str, Any]] = []  # List of conversation turns
        logger.info(f"Initialized ConversationMemory with max_history={max_history_length}, "
                   f"LLM provider: {llm_provider is not None}")

    def add_turn(self, question: str, answer: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        WHAT:  Append a question/answer turn to history and trim to max length.
        WHY:   Each turn becomes context for condensing future follow-ups; trimming
               bounds memory use in long conversations.
        INPUTS: question (str): the user's question; answer (str): the system's
                answer; metadata (Optional[Dict[str, Any]]): optional timestamp etc.
        OUTPUTS: None (mutates self.history in place).
        SIDE EFFECTS: Appends to self.history and may slice it to keep only the
                      most recent max_history_length turns (conversation_memory.py:49);
                      logs a debug line (conversation_memory.py:52).
        FAILURE MODES: metadata.get("timestamp") assumes metadata is a dict when
                       non-None — a non-dict metadata raises AttributeError
                       (conversation_memory.py:42).
        ALTERNATIVE: Store turns in a database with timestamps for durable,
                     queryable conversation history.
        """
        turn = {
            "question": question,
            "answer": answer,
            "timestamp": metadata.get("timestamp", datetime.now()) if metadata else datetime.now(),
            "metadata": metadata or {}
        }

        self.history.append(turn)

        # Keep only the most recent turns.
        if len(self.history) > self.max_history_length:
            self.history = self.history[-self.max_history_length:]

        logger.debug(f"Added conversation turn. History length: {len(self.history)}")

    def condense_question(self, current_question: str,
                          use_llm: bool = True) -> str:
        """
        WHAT:  Turn a follow-up question into a standalone query using history.
        WHY:   Follow-ups contain pronouns and references; retrieval needs a
               self-contained question to find the right chunks.
        INPUTS: current_question (str): the follow-up; use_llm (bool): whether to
                attempt LLM condensation (falls back to rules).
        OUTPUTS: str — a standalone query string.
        SIDE EFFECTS: None (read-only on history).
        FAILURE MODES: Empty history → returns the question unchanged (by design,
                       no context to condense with).
        ALTERNATIVE: Pass an already-built history via condense_follow_up_question()
                     instead of maintaining state in this instance.
        """
        if not self.history:
            # No history, return question as-is.
            return current_question

        if use_llm and self.llm_provider is not None:
            return self._condense_with_llm(current_question)
        else:
            return self._condense_with_rules(current_question)

    def get_standalone_query(self, current_question: str,
                             use_llm: bool = True) -> str:
        """
        WHAT:  Alias for condense_question() for clearer call-site naming.
        WHY:   Some callers read more naturally with "get_standalone_query";
               behaviour is identical to condense_question().
        INPUTS: current_question (str), use_llm (bool).
        OUTPUTS: str — standalone query (see condense_question()).
        SIDE EFFECTS: Delegates to condense_question().
        FAILURE MODES: Same as condense_question().
        ALTERNATIVE: Call condense_question() directly; this alias adds no logic.
        """
        return self.condense_question(current_question, use_llm)

    def _condense_with_llm(self, current_question: str) -> str:
        """
        WHAT:  Placeholder for LLM-based question condensation.
        WHY:   Documents where an LLM would be used; currently falls back to the
               rule-based condenser so behaviour is deterministic and testable.
        INPUTS: current_question (str): the follow-up question.
        OUTPUTS: str — condensed standalone query (from _condense_with_rules).
        SIDE EFFECTS: Logs an info line noting LLM would be used
                      (conversation_memory.py:106).
        FAILURE MODES: Exceptions caught, logged, and rule-based condensation used
                       (no exception propagates).
        ALTERNATIVE: Replace the placeholder with a real LLM call to
                     self.llm_provider using _build_conversation_context() as context.
        """
        try:
            # Build context from history.
            context = self._build_conversation_context()

            # In a real implementation, this would call the LLM provider.
            # For now, we'll use rule-based as fallback but indicate LLM would be used.
            logger.info(f"Would use LLM to condense question with context: {context[:100]}...")

            # Fall back to rule-based for now.
            return self._condense_with_rules(current_question)
        except Exception as e:
            logger.error(f"Error in LLM-based condensation: {e}")
            return self._condense_with_rules(current_question)

    def _condense_with_rules(self, current_question: str) -> str:
        """
        WHAT:  Condense a follow-up into a standalone question using regex patterns.
        WHY:   Detects follow-up openings and pronouns, then resolves references
               against the previous turn so retrieval gets a complete query.
        INPUTS: current_question (str): the follow-up question.
        OUTPUTS: str — condensed question, or the original if it is not a follow-up.
        SIDE EFFECTS: Logs a debug line when a follow-up is condensed
                      (conversation_memory.py:160).
        FAILURE MODES: Empty history → returns current_question (conversation_memory.py:124).
                       Pronoun replacement can produce awkward text when no
                       significant entities exist; a key-phrase fallback is applied.
        ALTERNATIVE: Use an LLM for more robust coreference resolution instead of
                     heuristic pronoun substitution.
        """
        if not self.history:
            return current_question

        # Get the most recent turn for context.
        last_turn = self.history[-1]
        last_question = last_turn["question"]
        last_answer = last_turn["answer"]

        current_lower = current_question.lower().strip()

        # Check for common follow-up patterns.
        follow_up_patterns = [
            r'^\s*(and|also|additionally|furthermore|moreover)\s+',
            r'^\s*(what|who|when|where|how|why)\s+(about|regarding|concerning)\s+',
            r'^\s*(tell me|give me|show me)\s+(more|additional)\s+',
            r'^\s*(however|but|although|though)\s+',
            r'^\s*(what about|how about)\s+',
            r'^\s*(and what|and how|and why)\s+',
            r'^\s*(is|are|was|were)\s+(it|this|that)\s+',
            r'^\s*(do|does|did)\s+(it|this|that)\s+',
            r'^\s*(can|could|would|should)\s+(it|this|that)\s+',
        ]

        is_follow_up = any(re.match(pattern, current_lower) for pattern in follow_up_patterns)

        # Also check for pronouns that refer to previous context.
        pronoun_patterns = [
            r'\b(it|this|that|they|them|their)\b',
            r'\b(he|she|his|her)\b',
        ]

        has_pronouns = any(re.search(pattern, current_lower) for pattern in pronoun_patterns)

        if is_follow_up or has_pronouns:
            # This appears to be a follow-up question, try to condense it.
            condensed = self._resolve_references(current_question, last_question, last_answer)
            logger.debug(f"Condensed follow-up: '{current_question}' -> '{condensed}'")
            return condensed
        else:
            # Doesn't appear to be a follow-up, return as-is.
            return current_question

    def _resolve_references(self, current_question: str,
                           last_question: str, last_answer: str) -> str:
        """
        WHAT:  Replace pronouns in the current question with entities from the
               previous Q&A turn.
        WHY:   "it", "this", "that" are meaningless to retrieval unless resolved
               to the concrete topic discussed in the last turn.
        INPUTS: current_question (str), last_question (str), last_answer (str).
        OUTPUTS: str — question with pronouns substituted by candidate entities.
        SIDE EFFECTS: None (pure string transformation).
        FAILURE MODES: Simplified entity extraction may substitute the wrong word
                       or leave text awkward; a key-phrase fallback prepends context
                       when the result is too short or unchanged.
        ALTERNATIVE: Use a proper NLP coreference resolver (spaCy, an LLM) for
                     accurate pronoun resolution.
        """
        # Extract key entities from last question and answer.
        # This is a simplified implementation - real NLP would be more sophisticated.

        condensed = current_question

        # Replace common pronouns with context from last turn.
        # Extract potential entities from last question (simplified).
        last_question_words = set(last_question.lower().split())
        last_answer_words = set(last_answer.lower().split())

        # Find significant words (not stop words) from last turn.
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}

        last_significant = (last_question_words | last_answer_words) - stop_words

        # If we have significant words, try to use them to resolve references.
        if last_significant:
            # Take the most significant words (longest ones often carry more meaning).
            sorted_significant = sorted(last_significant, key=len, reverse=True)
            top_entities = sorted_significant[:3]  # Top 3 entities

            # Replace pronouns with entities.
            condensed = re.sub(r'\bit\b', top_entities[0] if top_entities else 'it', condensed, flags=re.IGNORECASE)
            condensed = re.sub(r'\bthis\b', top_entities[0] if top_entities else 'this', condensed, flags=re.IGNORECASE)
            condensed = re.sub(r'\bthat\b', top_entities[0] if len(top_entities) > 1 else top_entities[0] if top_entities else 'that', condensed, flags=re.IGNORECASE)
            condensed = re.sub(r'\bthey\b', ' and '.join(top_entities[:2]) if len(top_entities) >= 2 else 'they', condensed, flags=re.IGNORECASE)

        # Clean up extra spaces.
        condensed = re.sub(r'\s+', ' ', condensed).strip()

        # If condensation didn't change much or resulted in something awkward, fall back.
        if len(condensed) < 3 or condensed == current_question.lower():
            # Try a different approach: prepend context from last question.
            # Extract key phrase from last question.
            last_key_phrase = self._extract_key_phrase(last_question)
            if last_key_phrase and len(last_key_phrase) > 3:
                condensed = f"{last_key_phrase} {current_question}"
                logger.debug(f"Used key phrase fallback: '{condensed}'")

        return condensed

    def _extract_key_phrase(self, text: str) -> str:
        """
        WHAT:  Pull a short topic phrase out of a question.
        WHY:   When pronoun substitution fails, prepending the last topic gives
               retrieval enough context to find relevant chunks.
        INPUTS: text (str): a previous question.
        OUTPUTS: str — up to 3 meaningful words, or a short fallback substring.
        SIDE EFFECTS: None (pure string transformation).
        FAILURE MODES: Returns a truncated substring when no meaningful words are
                       found; never raises.
        ALTERNATIVE: Use the LLM to summarize the prior turn into a context phrase.
        """
        # Remove question words and extract meaningful content.
        text_lower = text.lower()

        # Remove common question starters.
        question_starters = ['what is', 'what are', 'how does', 'how do', 'why is', 'why are',
                           'when is', 'when are', 'where is', 'where are', 'who is', 'who are']

        for starter in question_starters:
            if text_lower.startswith(starter):
                text_lower = text_lower[len(starter):].strip()
                break

        # Remove trailing question mark.
        text_lower = text_lower.rstrip('?')

        # Take first few meaningful words.
        words = [w for w in text_lower.split() if w not in {'the', 'a', 'an', 'and', 'or', 'but'}]
        if len(words) >= 3:
            return ' '.join(words[:3])
        elif len(words) > 0:
            return ' '.join(words)
        else:
            return text_lower[:20]  # Fallback to first 20 characters

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history.clear()
        logger.info("Conversation history cleared")

    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.history.copy()

    def _build_conversation_context(self) -> str:
        """
        WHAT:  Build a context string from the last few turns for an LLM.
        WHY:   An LLM condensing a follow-up needs to see the recent Q&A to
               understand what "it" or "that" refers to.
        INPUTS: None (uses self.history).
        OUTPUTS: str — formatted "Q: ... / A: ..." lines for the last 3 turns.
        SIDE EFFECTS: None (read-only on history).
        FAILURE MODES: Empty history → returns an empty string.
        ALTERNATIVE: Pass a sliding window of history directly to the LLM instead
                     of building a formatted string here.
        """
        context_parts = []
        for turn in self.history[-3:]:  # Last 3 turns
            context_parts.append(f"Q: {turn['question']}")
            context_parts.append(f"A: {turn['answer']}")
        return "\n".join(context_parts)


# Convenience functions
def create_conversation_memory(max_history_length: int = 10,
                             llm_provider: Any = None) -> ConversationMemory:
    """
    WHAT:  Build and return a ConversationMemory instance.
    WHY:   Factory convenience for the conversation memory.
    INPUTS: max_history_length (int): 10, llm_provider (Any): optional.
    OUTPUTS: ConversationMemory.
    SIDE EFFECTS: None.
    FAILURE MODES: None at construction.
    ALTERNATIVE: Instantiate ConversationMemory(...) directly.
    """
    return ConversationMemory(max_history_length, llm_provider)


def condense_follow_up_question(current_question: str,
                               conversation_history: List[Dict[str, Any]],
                               llm_provider: Any = None) -> str:
    """
    WHAT:  Condense a follow-up question using a supplied history list.
    WHY:   Lets callers condense without maintaining a ConversationMemory instance
               (e.g. when history comes from a database).
    INPUTS: current_question (str), conversation_history (List[Dict[str, Any]]),
            llm_provider (Any): optional.
    OUTPUTS: str — standalone query.
    SIDE EFFECTS: Constructs a ConversationMemory and copies the supplied history
                  into it (minor overhead).
    FAILURE MODES: Delegates to ConversationMemory.condense_question(); same behaviour.
    ALTERNATIVE: Keep a ConversationMemory instance if you condense many questions
                 with the same history.
    """
    memory = ConversationMemory(llm_provider=llm_provider)
    memory.history = conversation_history.copy()
    return memory.condense_question(current_question, llm_provider is not None)
