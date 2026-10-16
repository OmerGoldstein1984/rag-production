# Understanding RAG System Behaviors and Edge Cases

## Introduction
Retrieval-Augmented Generation (RAG) systems combine information retrieval with large language models to provide accurate, context-aware answers. While powerful, RAG systems exhibit specific behaviors and limitations that users should understand to set proper expectations and troubleshoot effectively. This document explores common RAG situations using practical examples from our Telecom RAG System.

## 1. Successful Retrieval - The Ideal Case

**Scenario:** The system finds relevant information and generates an accurate, well-cited answer.

**Example Query:** "What are the key benefits of eSIM technology for consumers?"

**Expected Behavior:**
- System retrieves relevant chunks from eSIM_document.txt
- Identifies key benefit statements about carrier switching, multiple profiles, roaming, and reduced SIM loss risk
- Generates answer that accurately synthesizes this information
- Provides proper citations pointing to the source document

**Actual Behavior (When Working Correctly):**
The system correctly identifies that eSIM benefits include: ability to switch carriers without changing physical SIM cards, support for multiple profiles on a single device, easier international roaming by downloading local carrier profiles, and reduced risk of SIM loss or damage. It presents this information clearly with appropriate citations to the eSIM document.

**Why This Happens:** When query terms match document content well and the retrieval system functions properly, the most relevant information is retrieved and the generation model can accurately synthesize it.

**Best Practice:** Use specific, descriptive queries that match the terminology used in your source documents for optimal retrieval performance.

## 2. Failed Retrieval - When Relevant Info Isn't Found

**Scenario:** Relevant information exists in the knowledge base but the system fails to retrieve it.

**Example Query:** "How does the eUICC security element prevent cloning attacks?"

**Expected Behavior:**
- System should retrieve information about eUICC (embedded Universal Integrated Circuit Card) security features from eSIM_document.txt
- Should find content about tamper resistance, secure element hardware certifications, and cloning attack prevention

**What Might Actually Happen:**
The system returns an "I don't know" response or provides a generic answer not specific to eUICC security features, even though the information exists in the documents.

**Why This Happens:**
- Query terminology mismatch: User says "eUICC security element" but documents might use "secure element" or "tamper-resistant hardware"
- Insufficient semantic similarity: The retrieval model doesn't connect the query concepts to the document content
- Poor chunking: Relevant information might be split across chunks in a way that reduces similarity scores
- Threshold filtering: Similarity scores might fall below the retrieval threshold

**How to Address:**
- Try alternative terminology: "How does the secure element in eSIM prevent cloning attacks?"
- Break complex questions into simpler parts
- Ensure documents contain sufficient detail on the topic
- Consider adjusting retrieval thresholds if consistently missing relevant information

## 3. Hallucination - When the System Invents Information

**Scenario:** The system generates information that sounds plausible but is not present in any source documents.

**Example Query:** "What is the maximum data transfer rate of eSIM technology according to 5G standards?"

**Expected Behavior:**
- System should respond that this information is not available in the provided documents
- Or state that eSIM technology itself doesn't have data transfer rates (it's a SIM technology, not a radio interface)

**What Might Actually Happen:**
The system generates a specific number like "eSIM technology supports data transfer rates up to 1 Gbps according to 5G specifications" - information that sounds technical and plausible but is completely fabricated.

**Why This Happens:**
- Language model overconfidence: The LLM tries to be helpful and generates specific information when uncertain
- Knowledge cutoff: The model's training data might contain related information it incorrectly applies
- Poor retrieval: If no relevant documents are found, some systems still attempt to generate an answer rather than admitting ignorance
- Lack of proper grounding: The generation phase isn't sufficiently constrained by retrieved content

**How to Address:**
- Always verify critical information against source documents
- Use systems that provide clear citations so you can check the source
- Be wary of overly specific numbers or claims without clear documentation
- Prefer systems that explicitly state when information is not available

## 4. Citation Issues - Problems with Attribution

**Scenario:** The system provides an answer but has problems with citation accuracy or completeness.

**Example Query:** "What security mechanisms protect SIM cards against cloning attacks?"

**Expected Behavior:**
- System retrieves information about tamper resistance, secure element certifications, and side-channel protection
- Provides answer citing the specific sections discussing these security mechanisms

**What Might Actually Happen:**
- **Missing Citations:** System provides accurate information about SIM security mechanisms but fails to cite any sources
- **Incorrect Citations:** System cites the wrong document or wrong section within a document
- **Over-Citation:** System cites numerous sources when only one or two are actually relevant
- **Under-Citation:** System cites sources but misses some that contain relevant information
- **Citation Format Issues:** Citations are malformed, incomplete, or difficult to trace back to source material

**Why This Happens:**
- Generation-Citation Mismatch: The answer generation and citation attribution processes don't align properly
- Retrieval Noise: Similar but irrelevant content gets cited alongside truly relevant information
- Post-Processing Errors: Citations get lost or corrupted during answer formatting
- Over-Attribution: System tries to be helpful by citing more sources than necessary
- Under-Attribution: System fails to track which retrieved chunks contributed to which parts of the answer

**How to Address:**
- Always check that citations actually support the claims made
- Verify you can locate the cited information in the source documents
- Report systematic citation issues to improve the system
- Use multiple information sources for critical fact-checking when possible

## 5. Edge Cases - Challenging Query Types

**Scenario:** Queries that push the limits of the RAG system's capabilities.

### Very Short Queries
**Example:** "eSIM benefits"
- **Issue:** Too ambiguous - could mean technical benefits, business benefits, consumer benefits, etc.
- **Result:** May retrieve irrelevant information or provide overly generic answers
- **Solution:** Add context: "What are the consumer benefits of eSIM technology?"

### Ambiguous Queries
**Example:** "Security improvements"
- **Issue:** Could refer to network security, application security, physical security, or SIM security
- **Result:** May retrieve mixed results from different security domains
- **Solution:** Specify context: "What security improvements does 5G AKA provide over previous versions?"

### Multi-Hop Questions
**Example:** "How does eSIM technology enable better IoT device management compared to traditional SIM cards in agricultural monitoring applications?"
- **Issue:** Requires connecting multiple concepts: eSIM benefits → IoT advantages → agricultural monitoring specifics
- **Result:** May answer only part of the question or miss the connections between concepts
- **Solution:** Break into simpler questions or accept that current RAG systems may struggle with complex logical chaining

### Temporal Questions
**Example:** "What was the original specification date for GSM SIM cards?"
- **Issue:** Requires specific historical information that may not be well-represented
- **Result:** May not find the information or provide inaccurate dates
- **Solution:** Include as much context as possible and verify any date information carefully

## 6. Best Practices for Getting Optimal Results

### Query Formulation
- **Be Specific:** Use precise terminology that matches your source documents
- **Provide Context:** Include relevant domain information to disambiguate queries
- **Use Complete Sentences:** Complete questions work better than keyword searches
- **One Concept at a Time:** For complex topics, ask simpler, focused questions

### System Understanding
- **Know Your Sources:** Familiarize yourself with what information is actually available
- **Check Coverage:** Understand which topics are well-covered vs. sparsely represented in your documents
- **Verify Critical Information:** Always double-check important facts against source materials
- **Understand Limitations:** Recognize that RAG systems retrieve and synthesize but don't "know" information like humans do

### Troubleshooting Approach
- **When You Get "I Don't Know":** Try reformulating with different terminology or broader context
- **When Answers Seem Wrong:** Check the cited sources - does the cited material actually support the claim?
- **When Getting Partial Answers:** Try breaking your question into simpler components
- **When Inconsistent Results:** Note that slight wording changes can sometimes yield different retrieval results

### Realistic Expectations
- **RAG is Not Omniscient:** The system can only work with information provided in the source documents
- **Quality Depends on Sources:** Better, more comprehensive source documents yield better results
- **It's About Assistance:** Think of RAG as a knowledgeable assistant that has read your documents, not an expert with infinite knowledge
- **Verification is Key:** Always treat RAG-generated information as helpful starting points, not final authority

## Conclusion
Understanding these common RAG situations helps users form realistic expectations and develop effective strategies for getting the most value from their RAG systems. By recognizing when the system is likely to succeed (clear queries matching well-represented topics) and when it might struggle (ambiguous queries, very specific niche information, or tasks requiring complex reasoning), users can adapt their approach accordingly. The key is to view the RAG system as a powerful tool for accessing and synthesizing documented information, while maintaining healthy skepticism and verification practices for critical information needs. With thoughtful query formulation and proper interpretation of results, users can significantly enhance their productivity and decision-making capabilities when working with RAG-enabled systems.