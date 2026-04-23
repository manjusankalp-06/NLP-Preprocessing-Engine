# Low-Level Design (LLD)

## RAG-Based Customer Support Assistant

---

## 1. Objective

The Low-Level Design (LLD) describes the internal implementation details of the RAG-based Customer Support Assistant. It explains how each module works, how data flows between components, and how decision-making is handled using LangGraph.

---

## 2. Module-Level Design

### 2.1 Document Processing Module

* Uses `PyPDFLoader` to load the PDF file
* Extracts raw text content from the knowledge base

---

### 2.2 Chunking Module

* Uses `RecursiveCharacterTextSplitter`
* Splits text into smaller chunks (e.g., 800 characters with overlap)
* Improves retrieval accuracy

---

### 2.3 Embedding Module

* Uses HuggingFace model: `all-MiniLM-L6-v2`
* Converts text chunks into vector embeddings
* These embeddings represent semantic meaning

---

### 2.4 Vector Storage Module

* Uses `ChromaDB`
* Stores embeddings along with metadata
* Supports similarity-based search

---

### 2.5 Retrieval Module

* Uses ChromaDB retriever (`k=4`)
* Fetches top relevant chunks based on query similarity
* Combines results into context

---

### 2.6 Query Processing Module

* Accepts user input
* Converts query into lowercase
* Detects intent (FAQ / escalation / general)

---

### 2.7 Graph Execution Module (LangGraph)

* Defines workflow using nodes:

  * `retrieve_node`
  * `answer_node`
  * `human_escalation_node`
* Controls execution flow
* Uses conditional routing

---

### 2.8 HITL Module (Human-in-the-Loop)

* Triggered when:

  * Confidence is low
  * Query is complex or sensitive
* Takes manual input from user (CLI)
* Overrides AI response

---

## 3. Data Structures

### 3.1 Document Representation

```text
Document:
- page_content: str
- metadata: dict
```

---

### 3.2 Chunk Format

```text
Chunk:
- text: str
- embedding: vector
```

---

### 3.3 Embedding Structure

* Numerical vector (list of floats)
* Represents semantic meaning of text

---

### 3.4 Query-Response Schema

```text
Input:
- user_query: str

Output:
- answer: str
- intent: str
- confidence: float
- escalate: bool
```

---

### 3.5 Graph State Object

```python
GraphState = {
    "user_query": str,
    "retrieved_context": str,
    "answer": str,
    "intent": str,
    "confidence": float,
    "escalate": bool,
    "escalation_reason": str,
    "human_response": str
}
```

---

## 4. Workflow Design (LangGraph)

### Nodes:

1. **Retrieve Node**

   * Retrieves relevant context from vector DB
   * Calculates confidence
   * Detects intent

2. **Answer Node**

   * Uses LLM (Gemini)
   * Generates response from context

3. **Human Escalation Node**

   * Takes human input
   * Returns manual response

---

### Edges:

* `retrieve → answer`
* `retrieve → human_escalation`

---

### State Flow:

```text
User Query → Retrieve Node → (Decision) → Answer / Human
```

---

## 5. Conditional Routing Logic

Routing is based on:

### Conditions:

* If confidence < 0.4 AND intent = escalation → escalate
* Otherwise → generate answer

```python
if confidence < 0.4 and intent == "escalation":
    route = "human_escalation"
else:
    route = "answer"
```

---

## 6. HITL Design

### When Triggered:

* Low confidence
* User is angry or escalates request

### Process:

1. Display query and context
2. Ask human to input response
3. Return human response as final output

---

### Benefits:

* Avoids hallucination
* Improves trust

---

## 7. API / Interface Design

### Input:

```text
User: What is refund policy?
```

---

### Output:

```text
Assistant: Refunds are processed within 5–7 business days.
Intent: faq
Confidence: 0.85
Escalated: No
```

---

### Interaction Flow:

1. User enters query
2. System processes via graph
3. Response displayed in CLI

---

## 8. Error Handling

### Case 1: Missing PDF

* Check file existence before loading

---

### Case 2: No Relevant Data

* Return fallback message:
  "Information not available in context"

---

### Case 3: LLM Failure

* Catch API errors
* Retry or fallback to HITL

---

## 9. Summary

The LLD explains how the system is implemented at a detailed level. It defines:

* Modules and responsibilities
* Data structures
* Workflow execution
* Decision logic

This design ensures a modular, scalable, and reliable AI system.

---
