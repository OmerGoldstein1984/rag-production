"""
Conversation memory utilities for condensing follow-up questions into standalone queries.
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
        Initialize the conversation memory.

        Args:
            max_history_length: Maximum number of conversation turns to remember
            llm_provider: LLM provider for query condensation (optional)
        """
        self.max_history_length = max_history_length
        self.llm_provider = llm_provider
        self.history: List[Dict[str, Any]] = []  # List of conversation turns
        logger.info(f"Initialized ConversationMemory with max_history={max_history_length}, "
                   f"LLM provider: {llm_provider is not None}")

    def add_turn(self, question: str, answer: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a conversation turn to memory.

        Args:
            question: User's question
            answer: System's answer
            metadata: Optional metadata (timestamp, etc.)
        """
        turn = {
            "question": question,
            "answer": answer,
            "timestamp": metadata.get("timestamp", datetime.now()) if metadata else datetime.now(),
            "metadata": metadata or {}
        }

        self.history.append(turn)

        # Keep only the most recent turns
        if len(self.history) > self.max_history_length:
            self.history = self.history[-self.max_history_length:]

        logger.debug(f"Added conversation turn. History length: {len(self.history)}")

    def condense_question(self, current_question: str,
                         use_llm: bool = True) -> str:
        """
        Condense a follow-up question into a standalone query using conversation history.

        Args:
            current_question: Current follow-up question
            use_llm: Whether to use LLM for condensation (fallback to rule-based if False)

        Returns:
            Standalone query string
        """
        if not self.history:
            # No history, return question as-is
            return current_question

        if use_llm and self.llm_provider is not None:
            return self._condense_with_llm(current_question)
        else:
            return self._condense_with_rules(current_question)

    def get_standalone_query(self, current_question: str,
                           use_llm: bool = True) -> str:
        """
        Get a standalone query from the current question and conversation history.
        Alias for condense_question for clarity.

        Args:
            current_question: Current follow-up question
            use_llm: Whether to use LLM for condensation

        Returns:
            Standalone query string
        """
        return self.condense_question(current_question, use_llm)

    def _condense_with_llm(self, current_question: str) -> str:
        """
        Condense question using LLM (placeholder - would call actual LLM).

        Args:
            current_question: Current follow-up question

        Returns:
            Condensed standalone query
        """
        try:
            # Build context from history
            context = self._build_conversation_context()

            # In a real implementation, this would call the LLM provider
            # For now, we'll use rule-based as fallback but indicate LLM would be used
            logger.info(f"Would use LLM to condense question with context: {context[:100]}...")

            # Fall back to rule-based for now
            return self._condense_with_rules(current_question)
        except Exception as e:
            logger.error(f"Error in LLM-based condensation: {e}")
            return self._condense_with_rules(current_question)

    def _condense_with_rules(self, current_question: str) -> str:
        """
        Condense question using rule-based approach.

        Args:
            current_question: Current follow-up question

        Returns:
            Condensed standalone query
        """
        if not self.history:
            return current_question

        # Get the most recent turn for context
        last_turn = self.history[-1]
        last_question = last_turn["question"]
        last_answer = last_turn["answer"]

        current_lower = current_question.lower().strip()

        # Check for common follow-up patterns
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

        # Also check for pronouns that refer to previous context
        pronoun_patterns = [
            r'\b(it|this|that|they|them|their)\b',
            r'\b(he|she|his|her)\b',
        ]

        has_pronouns = any(re.search(pattern, current_lower) for pattern in pronoun_patterns)

        if is_follow_up or has_pronouns:
            # This appears to be a follow-up question, try to condense it
            condensed = self._resolve_references(current_question, last_question, last_answer)
            logger.debug(f"Condensed follow-up: '{current_question}' -> '{condensed}'")
            return condensed
        else:
            # Doesn't appear to be a follow-up, return as-is
            return current_question

    def _resolve_references(self, current_question: str,
                           last_question: str, last_answer: str) -> str:
        """
        Resolve pronouns and references in current question using last Q&A.

        Args:
            current_question: Current follow-up question
            last_question: Previous question
            last_answer: Previous answer

        Returns:
            Question with resolved references
        """
        # Extract key entities from last question and answer
        # This is a simplified implementation - real NLP would be more sophisticated

        condensed = current_question

        # Replace common pronouns with context from last turn
        # Extract potential entities from last question (simplified)
        last_question_words = set(last_question.lower().split())
        last_answer_words = set(last_answer.lower().split())

        # Find significant words (not stop words) from last turn
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}

        last_significant = (last_question_words | last_answer_words) - stop_words

        # If we have significant words, try to use them to resolve references
        if last_significant:
            # Take the most significant words (longest ones often carry more meaning)
            sorted_significant = sorted(last_significant, key=len, reverse=True)
            top_entities = sorted_significant[:3]  # Top 3 entities

            # Replace pronouns with entities
            condensed = re.sub(r'\bit\b', top_entities[0] if top_entities else 'it', condensed, flags=re.IGNORECASE)
            condensed = re.sub(r'\bthis\b', top_entities[0] if top_entities else 'this', condensed, flags=re.IGNORECASE)
            condensed = re.sub(r'\bthat\b', top_entities[0] if len(top_entities) > 1 else top_entities[0] if top_entities else 'that', condensed, flags=re.IGNORECASE)
            condensed = re.sub(r'\bthey\b', ' and '.join(top_entities[:2]) if len(top_entities) >= 2 else 'they', condensed, flags=re.IGNORECASE)

        # Clean up extra spaces
        condensed = re.sub(r'\s+', ' ', condensed).strip()

        # If condensation didn't change much or resulted in something awkward, fall back
        if len(condensed) < 3 or condensed == current_question.lower():
            # Try a different approach: prepend context from last question
            # Extract key phrase from last question
            last_key_phrase = self._extract_key_phrase(last_question)
            if last_key_phrase and len(last_key_phrase) > 3:
                condensed = f"{last_key_phrase} {current_question}"
                logger.debug(f"Used key phrase fallback: '{condensed}'")

        return condensed

    def _extract_key_phrase(self, text: str) -> str:
        """
        Extract a key phrase from text that represents the main topic.

        Args:
            text: Input text

        Returns:
            Key phrase string
        """
        # Remove question words and extract meaningful content
        text_lower = text.lower()

        # Remove common question starters
        question_starters = ['what is', 'what are', 'how does', 'how do', 'why is', 'why are',
                           'when is', 'when are', 'where is', 'where are', 'who is', 'who are']

        for starter in question_starters:
            if text_lower.startswith(starter):
                text_lower = text_lower[len(starter):].strip()
                break

        # Remove trailing question mark
        text_lower = text_lower.rstrip('?')

        # Take first few meaningful words
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
        Build a context string from conversation history for LLM processing.

        Returns:
            Context string
        """
        context_parts = []
        for turn in self.history[-3:]:  # Last 3 turns
            context_parts.append(f"Q: {turn['question']}")
            context_parts.append(f"A: {turn['answer']}")
        return "\n".join(context_parts)


# Convenience functions
def create_conversation_memory(max_history_length: int = 10,
                             llm_provider: Any = None) -> ConversationMemory:
    """Create a conversation memory instance."""
    return ConversationMemory(max_history_length, llm_provider)


def condense_follow_up_question(current_question: str,
                               conversation_history: List[Dict[str, Any]],
                               llm_provider: Any = None) -> str:
    """
    Condense a follow-up question into a standalone query.

    Args:
        current_question: Current follow-up question
        conversation_history: List of previous conversation turns
        llm_provider: LLM provider for condensation (optional)

    Returns:
        Standalone query string
    """
    memory = ConversationMemory(llm_provider=llm_provider)
    memory.history = conversation_history.copy()
    return memory.condense_question(current_question, llm_provider is not None)