# High-Level Design (HLD)

## RAG-Based Customer Support Assistant

---

## 1. System Overview

### Problem Definition

Customer support systems often rely on manual responses, which leads to delays, inconsistency, and scalability issues. Users may not get quick answers to frequently asked questions (FAQs) such as refund policies, account issues, or product details.

### Proposed Solution

To address this, we designed a **RAG-based Customer Support Assistant** that:

* Retrieves relevant information from a PDF knowledge base
* Generates contextual answers using an LLM
* Uses a workflow system (LangGraph) for decision-making
* Escalates complex queries to a human (HITL)

---

## 2. System Architecture Diagram

```
           +-------------------+
           |       User        |
           +-------------------+
                     |
                     v
           +-------------------+
           |   Query Input     |
           +-------------------+
                     |
                     v
           +-------------------+
           |   LangGraph       |
           |  (Workflow Engine)|
           +-------------------+
              |            |
              |            v
              |     +-------------------+
              |     | Human-in-the-Loop|
              |     |   (Escalation)    |
              |     +-------------------+
              |
              v
    +--------------------------+
    | Retrieval System         |
    | (ChromaDB + Retriever)   |
    +--------------------------+
              |
              v
    +--------------------------+
    | LLM (Gemini API)         |
    +--------------------------+
              |
              v
    +--------------------------+
    |   Final Response         |
    +--------------------------+
```

---

## 3. System Components

### 3.1 Document Loader

* Loads the PDF knowledge base using `PyPDFLoader`
* Extracts raw text from the document

### 3.2 Chunking Strategy

* Uses `RecursiveCharacterTextSplitter`
* Splits large text into smaller chunks (for better retrieval)

### 3.3 Embedding Model

* Uses HuggingFace embeddings (`all-MiniLM-L6-v2`)
* Converts text chunks into numerical vectors

### 3.4 Vector Store (ChromaDB)

* Stores embeddings efficiently
* Enables fast semantic search

### 3.5 Retriever

* Fetches relevant chunks based on user query
* Uses similarity search (top-k results)

### 3.6 LLM (Large Language Model)

* Uses Gemini API (`gemini-flash-latest`)
* Generates human-like responses based on retrieved context

### 3.7 Workflow Engine (LangGraph)

* Controls the flow of the system
* Defines nodes like:

  * Retrieval Node
  * Answer Node
  * Human Escalation Node

### 3.8 Routing Layer

* Decides:

  * Answer directly
  * OR escalate to human

### 3.9 Human-in-the-Loop (HITL)

* Handles:

  * Low confidence queries
  * Sensitive or complex issues
* Allows human intervention instead of wrong AI answers

---

## 4. Data Flow

### Step-by-Step Flow

1. User enters query
2. Query is passed to LangGraph
3. Retrieval node fetches relevant data from ChromaDB
4. System checks:

   * Intent
   * Confidence
5. If confidence is high → LLM generates answer
6. If confidence is low → query is escalated to human
7. Final response is returned to user

---

## 5. Technology Choices

### ChromaDB

* Lightweight vector database
* Easy integration with LangChain
* Efficient semantic search

### LangGraph

* Enables workflow-based AI systems
* Supports conditional routing
* Helps manage decision logic

### Gemini API

* Fast and efficient LLM
* Generates accurate responses
* Supports real-time applications

### Python

* Easy to implement
* Rich ecosystem for AI/ML tools

---

## 6. Scalability Considerations

### Handling Large Documents

* Use chunking to break large PDFs
* Store efficiently in vector database

### Increasing Query Load

* Use optimized retrieval (top-k search)
* Scale vector database if needed

### Latency Optimization

* Use lightweight embedding models
* Use fast LLM (Gemini Flash)

---

## 7. Summary

This system is not just a chatbot but a **structured AI system** that:

* Combines retrieval + generation
* Uses workflow-based decision making
* Ensures reliability using HITL

It is designed to be scalable, accurate, and safe for real-world customer support applications.

---
