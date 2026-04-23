# Technical Documentation
## RAG-Based Customer Support Assistant using LangGraph & HITL

### 1. Introduction

#### 1.1 What is RAG?
Retrieval-Augmented Generation (RAG) is an architecture that combines a retrieval system over external knowledge (e.g., documents, databases) with a generative language model. Instead of answering purely from its internal parameters, the model retrieves relevant documents and grounds its answer in that retrieved evidence.[web:16][web:20]

#### 1.2 Why RAG is Needed
Traditional chatbots either rely on keyword search or purely on the LLM’s internal memory. Both approaches can lead to:
- Outdated information.
- Hallucinations (confident but wrong answers).
- Difficulty enforcing company-specific policies.  
RAG solves this by:
- Using an updatable knowledge base (PDFs, docs).
- Conditioning the LLM on retrieved chunks from this knowledge base.
- Allowing knowledge updates without retraining the LLM.[web:16][web:20]

#### 1.3 Use Case Overview
This project implements a RAG-based **Customer Support Assistant** that:
- Ingests a PDF knowledge base (e.g., FAQ or policy document).
- Answers user questions from this knowledge base.
- Uses a workflow engine (LangGraph) for explicit control over steps.
- Escalates to a human agent (HITL) when the model is uncertain or when the query is sensitive.[cite:2][web:15][web:21][web:23]

---

### 2. System Architecture Explanation

#### 2.1 HLD Summary
At a high level, the system consists of:
- CLI user interface.
- PDF ingestion and chunking.
- Embeddings + ChromaDB vector store.
- Retrieval module.
- LLM-based answer generation.
- LangGraph workflow for routing and state.
- HITL module for human escalation.[cite:8][web:9][web:10][web:19][web:20]

#### 2.2 Component Interactions
1. **Startup**
   - User provides PDF path.
   - PDF is loaded and chunked.
   - Chunks are embedded and stored in ChromaDB.
2. **Query Handling**
   - User enters a query in CLI.
   - LangGraph initializes state with this query.
   - Process node:
     - Retrieves top-k chunks from ChromaDB.
     - Calls LLM to generate an answer using retrieved context.
     - Scores the answer and decides route.
   - Output node:
     - Auto-answers or triggers HITL.
3. **HITL Flow**
   - Human sees query + context + draft answer.
   - Human approves or edits answer.
   - Final answer is returned to user.

---

### 3. Design Decisions

#### 3.1 Chunk Size Choice
- Chunk size ≈ 800–1000 characters with overlap ≈ 150–200:
  - Large enough to capture full paragraphs or small sections.
  - Overlap preserves continuity across boundaries.
- Trade-off:
  - Smaller chunks → better retrieval precision but more DB entries.
  - Larger chunks → fewer DB entries but risk mixing topics.[web:16][web:20]

#### 3.2 Embedding Strategy
- Use a single sentence embedding model for both documents and queries:
  - Ensures they live in the same vector space.
  - Enables accurate similarity search.[web:14][web:20][web:22]
- Embeddings are computed once during ingestion and stored in ChromaDB.

#### 3.3 Retrieval Approach
- Similarity search over ChromaDB:
  - Use cosine similarity or inner product ranking.[web:20][web:22]
  - Retrieve top-k chunks (e.g., k=4 or 6).
- Optionally apply:
  - Threshold on similarity score.
  - Simple filters on metadata (e.g., source document or section).

#### 3.4 Prompt Design Logic
- System prompt:
  - “You are a helpful customer support assistant. Use only the provided context to answer. If the context is insufficient, say you are not sure and suggest escalation.”
- Prompt structure:
  1. System message with instructions.
  2. Context section:
     - Each chunk with metadata (page numbers, sections).
  3. User query:
     - Natural language question.
- This structure encourages grounded answers and makes it easier to detect low confidence.

---

### 4. Workflow Explanation (LangGraph)

#### 4.1 LangGraph Basics
- LangGraph is used to define a stateful graph of nodes (steps) and edges (transitions):
  - `StateGraph` holds the state type.
  - Nodes transform the state.
  - Edges connect nodes from START to END with possible branching.[web:9][web:10][web:19]

#### 4.2 Node Responsibilities
- **process_node**
  - Input: `GraphState` with `query`.
  - Responsibilities:
    - Retrieve relevant chunks from ChromaDB.
    - Generate model answer with LLM.
    - Compute `confidence` score.
    - Set `route` to `"answer"` or `"hitl"`.
- **output_node**
  - Input: `GraphState` with `route` and `model_answer`.
  - Responsibilities:
    - If `route == "answer"`:
      - Set `final_answer = model_answer`.
    - If `route == "hitl"`:
      - Invoke HITL module to collect a human-approved answer.

#### 4.3 State Transitions
1. Initial state:
   ```python
   {"query": user_query}
   ```
2. After `process_node`:
   ```python
   {
       "query": user_query,
       "context_chunks": [...],
       "model_answer": "...",
       "confidence": 0.0–1.0,
       "route": "answer" or "hitl",
       "needs_escalation": True/False
   }
   ```
3. After `output_node`:
   ```python
   {
       "query": user_query,
       "context_chunks": [...],
       "model_answer": "...",
       "final_answer": "...",
       "route": "answer" or "hitl",
       "confidence": 0.0–1.0,
       "needs_escalation": False
   }
   ```

---

### 5. Conditional Logic

#### 5.1 Intent Detection
Intent can be approximated using:
- Simple keyword-based rules:
  - Billing-related words, legal terms, medical terms.
- Or by asking the LLM:
  - “Classify the user intent as one of: GENERAL_SUPPORT, BILLING, LEGAL, OUT_OF_SCOPE.”

Sensitive intents (e.g., LEGAL, MEDICAL, HIGH-RISK BILLING) increase the probability of escalation.[web:15][web:21][web:23]

#### 5.2 Routing Decisions
Routing combines:
- Similarity scores from retrieval.
- Answer confidence from scoring.
- Intent from classification.
- Explicit user requests for human agent.

Example:
- If:
  - `max_similarity_score >= 0.7`
  - Intent is in-scope.
  - No sensitive keywords.
- Then:
  - `route = "answer"`.
- Else:
  - `route = "hitl"`.

---

### 6. HITL Implementation

#### 6.1 Role of Human Intervention
Human agents handle:
- Ambiguous queries.
- Out-of-scope or policy-sensitive cases.
- Situations where the model expresses uncertainty.

HITL ensures:
- Higher reliability and safety.
- Better handling of edge cases and exceptions.[web:15][web:18][web:21][web:23]

#### 6.2 HITL Workflow
1. System detects low confidence or sensitive intent.
2. `route = "hitl"`, `needs_escalation = True`.
3. CLI displays:
   - Query.
   - Retrieved context chunks.
   - Model draft answer.
4. Human agent:
   - Approves model answer.
   - Edits or rewrites answer.
   - Requests more info from user.
5. Final answer is stored in `final_answer` and returned to user.

#### 6.3 Benefits and Limitations
- **Benefits**
  - Reduces risk of incorrect or unsafe answers.
  - Allows gradual automation while maintaining human oversight.
- **Limitations**
  - Requires human agents to be available.
  - Increases latency for escalated cases.
  - Needs good training and tooling for agents.

---

### 7. Challenges & Trade-offs

#### 7.1 Retrieval Accuracy vs Speed
- Higher `k` and more complex re-ranking improves accuracy but increases compute time.
- Pre-filtering by section or metadata can improve speed at the cost of some recall.[web:20][web:22]

#### 7.2 Chunk Size vs Context Quality
- Larger chunks:
  - Pros: richer context, fewer DB entries.
  - Cons: may mix unrelated content, making answers less precise.
- Smaller chunks:
  - Pros: more precise retrieval.
  - Cons: need more chunks to reconstruct full answer, may increase prompt length.[web:16][web:20]

#### 7.3 Cost vs Performance
- Larger LLMs give better reasoning and fluency but are more expensive and slower.
- Strategy:
  - Use a small LLM for routing and intent detection.
  - Use a larger LLM only when needed for complex questions.[web:12][web:21]

---

### 8. Testing Strategy

#### 8.1 Testing Approach
- **Unit Tests**
  - PDF loading and chunking.
  - Embedding and retrieval logic (using fixed embeddings).
- **Integration Tests**
  - End-to-end flow:
    - Query → Retrieval → LLM → Routing.
  - Simulate low-confidence scenarios to validate HITL routing.
- **Manual Testing**
  - Ask typical support queries:
    - “What is the refund policy?”
    - “How long does shipping take?”
  - Ask sensitive or out-of-scope queries:
    - “Can you give legal advice about my contract?”
    - “What is my credit card number?” (should not answer).

#### 8.2 Sample Queries
- In-scope:
  - “What are your working hours?”
  - “How can I reset my password?”
- Sensitive / escalation:
  - “Can you approve a refund above the standard limit?”
  - “Please give me legal advice about this clause.”

---

### 9. Future Enhancements

#### 9.1 Multi-Document Support
- Extend ingestion pipeline to handle multiple PDFs:
  - Tag chunks with `document_id`.
  - Allow filtering retrieval by document type or category.

#### 9.2 Feedback Loop
- Log all queries, answers, and human overrides.
- Use this data to:
  - Improve routing thresholds.
  - Identify gaps in the knowledge base.
  - Fine-tune prompts or models.

#### 9.3 Memory Integration
- Add short-term conversation memory:
  - Track previous turns.
  - Provide conversational context to LLM.

#### 9.4 Deployment
- Wrap CLI logic into a REST API (FastAPI / Flask).
- Add a web-based UI for both end users and human agents.
- Containerize using Docker for easy deployment to cloud platforms.

---

### 10. Conclusion
This technical documentation describes the end-to-end design and behavior of a RAG-based Customer Support Assistant using LangGraph, ChromaDB, and a PDF knowledge base. The system is designed to be modular, explainable, and safe, using Human-in-the-Loop escalation to handle complex or low-confidence cases while still benefiting from AI automation.