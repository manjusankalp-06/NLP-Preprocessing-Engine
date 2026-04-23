# High-Level Design (HLD)
## RAG-Based Customer Support Assistant using LangGraph & HITL

### 1. System Overview

#### 1.1 Problem Definition
Modern customer support teams handle large volumes of queries that are mostly answered by existing documentation such as FAQs, product manuals, and policy PDFs. Manually searching these documents is slow and error-prone.  
This project designs a Retrieval-Augmented Generation (RAG) based Customer Support Assistant that can automatically answer user questions from a PDF knowledge base while safely escalating complex or low-confidence cases to a human agent (Human-in-the-Loop, HITL).[cite:2][web:16][web:20]

#### 1.2 Scope of the System
- Single PDF knowledge base (for example, `sample_kb.pdf`) serves as the primary source of truth.[cite:2]
- Command Line Interface (CLI) based assistant (`python app.py`).
- RAG pipeline:
  - Load PDF → Chunk text → Generate embeddings → Store in ChromaDB → Retrieve relevant chunks → Use LLM to generate answer.[cite:2][web:16][web:20][web:22]
- Graph-based control flow using LangGraph:
  - Explicit nodes and edges for RAG processing and decision-making.[web:9][web:10][web:19]
- Intent and confidence based routing:
  - Auto-answer vs escalate to HITL.
- Human-in-the-Loop (HITL) escalation:
  - Human reviews or overrides the model’s answer.

---

### 2. Architecture Overview

#### 2.1 Architecture Diagram (Text Description)

The architecture consists of the following major blocks:

1. **User Interface (CLI)**
   - User types natural language queries and sees responses.
   - System shows whether the answer is auto-generated or escalated to a human.

2. **Document Ingestion Pipeline**
   - PDF Loader (PyPDFLoader) reads the knowledge base PDF.
   - Text cleaning and normalization.
   - Chunking using RecursiveCharacterTextSplitter with configured chunk size and overlap.

3. **Embedding System**
   - Embedding model (e.g., HuggingFace or OpenAI embeddings) converts text chunks into vectors.[web:14][web:20][web:22]
   - Shared embedding function is also used to embed user queries.

4. **Vector Database (ChromaDB)**
   - Stores chunk text, metadata, and embeddings in a Chroma collection (e.g., `customer_support_kb`).[web:20][web:22]
   - Supports similarity search to retrieve top-k relevant chunks.

5. **Retrieval Layer**
   - Given a user query, computes query embedding and queries ChromaDB.
   - Returns top-k context chunks and associated metadata.

6. **LLM Processing Layer**
   - Large Language Model (LLM) generates an answer using:
     - User query.
     - Retrieved context chunks.
   - Also optionally used for intent detection or answer confidence scoring.[web:12]

7. **Workflow Orchestration (LangGraph)**
   - LangGraph `StateGraph` orchestrates the flow:
     - State includes fields such as: `query`, `context_chunks`, `model_answer`, `confidence`, `route`, `needs_escalation`.[web:9][web:10][web:12][web:19]
   - Nodes:
     - `process_node`: performs retrieval, answer generation, and routing decision.
     - `output_node`: produces the final answer or triggers HITL.
   - Conditional edges route between nodes based on intent and confidence.

8. **Routing Layer**
   - Uses heuristic or LLM-based rules to decide:
     - When to auto-answer.
     - When to escalate to human.
   - Factors: similarity scores, query intent, model uncertainty, and sensitive topics.[web:12][web:21][web:23]

9. **HITL System**
   - Triggered when `needs_escalation` is true.
   - Human sees the query, retrieved context, and model draft answer in the CLI.
   - Human can approve, edit, or fully override the answer.[web:15][web:18][web:21][web:23]

High-level text diagram:

User (CLI)  
→ LangGraph (Input / Process node)  
→ Retrieval (Embeddings + ChromaDB)  
→ LLM Answer Generation  
→ Routing (Decision)  
→ Auto Answer OR HITL Escalation  
→ Output back to User

---

### 3. Component Description

#### 3.1 Document Loader
- Uses PyPDFLoader to load the PDF knowledge base specified by the user (e.g., `sample_kb.pdf`).[cite:7]
- Reads all pages into in-memory representations (one per page).

#### 3.2 Chunking Strategy
- Uses RecursiveCharacterTextSplitter to chunk pages into overlapping text segments:
  - Example: chunk_size ≈ 800–1000 characters.
  - chunk_overlap ≈ 150–200 characters.[web:16][web:20]
- Each chunk includes metadata such as page number and source filename.

#### 3.3 Embedding Model
- Uses a text embedding function (e.g., HuggingFace or OpenAI) to convert chunks and queries into dense vectors.[web:14][web:20][web:22]
- Same embedding model is used for both documents and queries.

#### 3.4 Vector Store (ChromaDB)
- A ChromaDB collection (e.g., `customer_support_kb`) persists:
  - `documents` (chunk text).
  - `metadatas` (page, source).
  - `embeddings` (vector representation).[web:20][web:22]
- Supports similarity search for retrieval.

#### 3.5 Retriever
- Given a text query:
  - Embeds the query.
  - Performs similarity search in ChromaDB.
  - Returns top-k relevant chunks and scores.

#### 3.6 LLM
- Chat-oriented LLM used for:
  - Answer generation using retrieved chunks as context.
  - Optional intent classification or confidence estimation.[web:12]

#### 3.7 Graph Workflow Engine (LangGraph)
- LangGraph `StateGraph` holds the conversation state.
- Nodes:
  - `process_node`:
    - Retrieve context.
    - Generate model answer.
    - Compute confidence and routing decision.
  - `output_node`:
    - Route to auto-answer or HITL.
- Edges:
  - `START → process_node → output_node` with conditional routing.[web:9][web:10][web:19]

#### 3.8 Routing Layer
- Uses rules to classify each query:
  - In-scope vs out-of-scope.
  - Sensitive vs non-sensitive.
  - High vs low confidence.
- Sets `route` to `"answer"` or `"hitl"`.

#### 3.9 HITL Module
- Invoked when `route == "hitl"` or `needs_escalation == True`.
- Prompts human agent in CLI to:
  - Approve model answer.
  - Edit/override answer.
  - Ask user for further details.[web:15][web:18][web:21][web:23]

---

### 4. Data Flow

#### 4.1 From PDF to Vector Store
1. User starts app and enters PDF path.
2. Document loader reads the PDF into page objects.[cite:7]
3. Chunking module splits text into overlapping chunks.
4. Embedding module converts each chunk into a vector.
5. Vector store (ChromaDB) stores chunks + embeddings in a persistent collection.[web:20][web:22]

#### 4.2 Query Lifecycle
1. User enters a natural language query in CLI.
2. LangGraph initializes state with `query` and other default fields.
3. `process_node`:
   - Embeds query.
   - Retrieves top-k chunks from ChromaDB.
   - Calls LLM with query + retrieved context.
   - Computes confidence and intent.
   - Sets `route` and `needs_escalation`.
4. `output_node`:
   - If route is `"answer"`:
     - Returns model answer directly to user.
   - If route is `"hitl"`:
     - Engages HITL module for human review.
5. Final answer (model or human) is printed back to the CLI.

---

### 5. Technology Choices

#### 5.1 ChromaDB
- Chosen as the vector database because:
  - Lightweight, easy to run locally.
  - Good Python integration with LangChain.
  - Persistent storage for embeddings and metadata.[web:20][web:22]

#### 5.2 LangGraph
- Provides a graph-based abstraction over the RAG workflow:
  - Makes nodes, edges, and state transitions explicit.
  - Fits the requirement of designing a workflow-based AI system with conditional routing.[web:9][web:10][web:19]

#### 5.3 LLM Choice
- Any chat-capable LLM accessible from Python (e.g., OpenAI, Groq, or local model):
  - Handles natural language understanding and answer generation.
  - Supports structured output for intent and confidence when required.[web:12]

#### 5.4 Additional Tools
- PyPDFLoader for PDF ingest.
- RecursiveCharacterTextSplitter for chunking.
- Logging for debugging and observability.

---

### 6. Scalability Considerations

#### 6.1 Handling Large Documents
- Use streaming or page-by-page loading for very large PDFs.
- Precompute and persist embeddings once at startup to avoid recomputation.[web:16][web:20]
- Consider hierarchical chunking if document is very large (chapters → sections → paragraphs).

#### 6.2 Increasing Query Load
- Wrap CLI logic into a web API (e.g., FastAPI) to support concurrent clients.
- Keep ChromaDB process warm with cached connections.
- Use horizontal scaling for API layer.

#### 6.3 Latency Concerns
- Cache popular queries and responses.
- Use smaller, faster models for initial routing and confidence scoring, and reserve larger models for complex queries.[web:12][web:21]
- Tune chunk size and top-k retrieval to reduce context length.

---

### 7. Summary
This HLD describes a RAG-based Customer Support Assistant that processes a PDF knowledge base, stores embeddings in ChromaDB, uses LangGraph for workflow control, and integrates Human-in-the-Loop escalation for safe and reliable answers. The system is designed to be modular, explainable, and production-oriented.