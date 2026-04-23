# Technical Documentation

## RAG-Based Customer Support Assistant

---

## 1. Introduction

### What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that combines:

* **Retrieval** → Fetch relevant information from a knowledge base
* **Generation** → Use an LLM to generate answers based on that information

---

### Why RAG is Needed

Traditional chatbots rely only on pre-trained knowledge and may generate incorrect or outdated answers (hallucination).
RAG solves this by grounding responses in real documents.

---

### Use Case Overview

This project implements a **Customer Support Assistant** that:

* Answers queries from a PDF knowledge base
* Provides accurate, context-aware responses
* Escalates complex queries to a human

---

## 2. System Architecture Explanation

The system follows a pipeline-based architecture:

1. PDF is loaded and processed
2. Text is split into chunks
3. Chunks are converted into embeddings
4. Stored in a vector database (ChromaDB)
5. User query is processed
6. Relevant context is retrieved
7. LLM generates an answer
8. If needed, query is escalated to human

---

### Component Interaction

* **PyPDFLoader** → Loads document
* **Text Splitter** → Breaks text into chunks
* **Embeddings** → Convert text into vectors
* **ChromaDB** → Stores vectors
* **Retriever** → Fetches relevant data
* **LangGraph** → Controls workflow
* **LLM (Gemini)** → Generates response
* **HITL Module** → Handles escalation

---

## 3. Design Decisions

### Chunk Size Selection

* Chunk size = ~800 characters
* Overlap = ~150 characters
* Reason: Balance between context and retrieval accuracy

---

### Embedding Strategy

* Model: `all-MiniLM-L6-v2`
* Reason:

  * Lightweight
  * Fast
  * Good semantic understanding

---

### Retrieval Approach

* Top-k similarity search (k=4)
* Ensures relevant context is retrieved

---

### Prompt Design Logic

* Provide:

  * User query
  * Retrieved context
* Restrict LLM:

  * “Use only provided context”
* Prevent hallucination

---

## 4. Workflow Explanation (LangGraph)

LangGraph is used to manage the system flow.

---

### Nodes

1. **Retrieve Node**

   * Fetches context
   * Detects intent
   * Calculates confidence

2. **Answer Node**

   * Uses LLM to generate response

3. **Human Escalation Node**

   * Handles manual input

---

### State Transitions

```text id="xq2u8q"
User Query → Retrieve Node → (Decision) → Answer Node / HITL Node
```

---

## 5. Conditional Logic

### Intent Detection

* Keywords used:

  * FAQ → “what”, “how”, “policy”
  * Escalation → “angry”, “complaint”

---

### Routing Decisions

```text id="i3b4ux"
If confidence < threshold AND intent = escalation → HITL  
Else → Answer Node
```

---

## 6. Human-in-the-Loop (HITL)

### Role

* Provides fallback when AI is uncertain
* Ensures reliability

---

### Benefits

* Avoids incorrect answers
* Improves trust

---

### Limitations

* Requires human availability
* Slower response

---

## 7. Challenges & Trade-offs

### 1. Retrieval Accuracy vs Speed

* More chunks → better accuracy
* But slower performance

---

### 2. Chunk Size vs Context Quality

* Large chunks → better context
* Small chunks → better retrieval

---

### 3. Cost vs Performance

* Larger models → better answers
* But higher cost

---

## 8. Testing Strategy

### Approach

* Test with multiple queries:

  * FAQ-based
  * Unknown queries
  * Escalation queries

---

### Example Queries

* “What is refund policy?”
* “How to reset password?”
* “I am unhappy with service”

---

## 9. Future Enhancements

* Multi-document support
* Web UI using Streamlit
* API deployment
* Feedback learning system
* Chat history memory

---

## 10. Conclusion

This project demonstrates how to build a **real-world AI system** using:

* Retrieval (ChromaDB)
* Generation (LLM)
* Workflow control (LangGraph)
* Human fallback (HITL)

It ensures accuracy, reliability, and scalability, making it suitable for modern customer support applications.

---
