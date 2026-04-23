# Low-Level Design (LLD)
## RAG-Based Customer Support Assistant using LangGraph & HITL

### 1. Module-Level Design

The implementation is centered around `app.py`, organized into logical modules (within a single file or split into future packages).[cite:2][cite:8]

#### 1.1 Document Processing Module
- **Responsibilities**
  - Load and parse a PDF file from a user-provided path.
  - Normalize/clean extracted text as needed.
- **Key Functions**
  - `load_pdf(pdf_path: str) -> List[DocumentPage]`
    - Uses `PyPDFLoader(pdf_path).load()`.
  - `clean_text(text: str) -> str`
    - Optional; removes excessive whitespace, control characters, etc.[cite:7]

#### 1.2 Chunking Module
- **Responsibilities**
  - Split document pages into overlapping text chunks suitable for embeddings.
- **Key Functions**
  - `chunk_documents(pages: List[DocumentPage], chunk_size: int, overlap: int) -> List[Chunk]`
- **Details**
  - Uses `RecursiveCharacterTextSplitter` with:
    - `chunk_size ≈ 800–1000` characters.
    - `chunk_overlap ≈ 150–200` characters.[web:16][web:20]
  - Each chunk stores metadata:
    - `{"page": page_number, "source": pdf_file_name}`.

#### 1.3 Embedding Module
- **Responsibilities**
  - Provide embeddings for chunks and queries.
- **Key Functions**
  - `get_embeddings_model() -> EmbeddingFunction`
  - `embed_chunks(chunks: List[Chunk]) -> List[EmbeddingVector]` (if precomputing).
- **Details**
  - Uses a sentence embedding model (HuggingFace / OpenAI) configured via environment variables or code.[web:14][web:20][web:22]
  - Integrated with ChromaDB either:
    - by passing an embedding function to Chroma, or
    - by precomputing embeddings and passing them explicitly.

#### 1.4 Vector Storage Module
- **Responsibilities**
  - Initialize and manage the ChromaDB collection.
  - Persist and query embeddings.
- **Key Functions**
  - `init_vector_store(persist_dir: str) -> ChromaCollection`
  - `store_chunks(collection, chunks: List[Chunk]) -> None`
  - `query_collection(collection, query: str, k: int) -> List[Chunk]`
- **Details**
  - Uses Chroma’s Python API to:
    - Create/get a collection.
    - Add:
      - `documents = [chunk.text]`
      - `metadatas = [chunk.metadata]`
      - `ids = [chunk.id]`.[web:20][web:22]

#### 1.5 Retrieval Module
- **Responsibilities**
  - Encapsulate retrieval logic for user queries.
- **Key Functions**
  - `retrieve_context(collection, query: str, k: int) -> List[Chunk]`
- **Details**
  - Embeds the query using the same embedding model.
  - Calls Chroma’s `similarity_search` or equivalent.
  - Returns sorted chunks and similarity scores.

#### 1.6 Query Processing Module
- **Responsibilities**
  - Prepare prompts.
  - Call the LLM for answer generation.
  - Compute confidence and route decisions.
- **Key Functions**
  - `generate_answer(query: str, chunks: List[Chunk]) -> str`
  - `score_answer(query: str, chunks: List[Chunk], answer: str) -> float`
  - `route_decision(query: str, chunks: List[Chunk], answer: str, score: float) -> str`
- **Details**
  - Prompt includes:
    - System instruction.
    - User query.
    - Context chunks with metadata.
  - `score_answer`:
    - May use heuristic (length, keywords, similarity scores).
    - Or call a secondary LLM “judge” for confidence scoring.[web:12]

#### 1.7 Graph Execution Module (LangGraph)
- **Responsibilities**
  - Define and execute the LangGraph state machine.
- **State**
  ```python
  class Chunk(TypedDict):
      id: str
      text: str
      metadata: dict

  class GraphState(TypedDict, total=False):
      query: str
      context_chunks: List[Chunk]
      model_answer: str
      final_answer: str
      route: Literal["answer", "hitl", "end"]
      confidence: float
      needs_escalation: bool
  ```
- **Nodes**
  - `process_node(state: GraphState) -> GraphState`
    - Performs retrieval, calls `generate_answer`, computes `confidence`, and sets `route` & `needs_escalation`.
  - `output_node(state: GraphState) -> GraphState`
    - If `route == "answer"`:
      - `final_answer = model_answer`.
    - If `route == "hitl"`:
      - Calls HITL module and sets `final_answer`.
- **Graph**
  ```python
  graph = StateGraph(GraphState)
  graph.add_node("process", process_node)
  graph.add_node("output", output_node)
  graph.add_edge(START, "process")
  graph.add_edge("process", "output")
  app = graph.compile()
  ```
  - Additional conditional edges can be added if needed.[web:9][web:10][web:19]

#### 1.8 HITL Module
- **Responsibilities**
  - Handle escalation to human-in-the-loop.
- **Key Function**
  - `hitl_escalate(state: GraphState) -> GraphState`
- **Details**
  - In CLI:
    - Show:
      - User query.
      - Retrieved context (selected chunks).
      - Model draft answer.
    - Ask agent:
      1. Approve model answer.
      2. Edit/override answer.
      3. Ask for clarification from user.
  - Updates `final_answer` and sets `needs_escalation = False`.

---

### 2. Data Structures

#### 2.1 Document Representation
```python
class DocumentPage(TypedDict):
    page_number: int
    text: str
```

#### 2.2 Chunk Format
```python
class Chunk(TypedDict):
    id: str
    text: str
    metadata: dict  # {"page": int, "source": str, "section": Optional[str]}
```

#### 2.3 Embedding Structure
```python
EmbeddingVector = List[float]
```
- Dimension depends on embedding model (e.g., 768 or 1536).[web:14][web:20][web:22]

#### 2.4 Query-Response Schema
```python
class QAInteraction(TypedDict, total=False):
    query: str
    context_chunks: List[Chunk]
    model_answer: str
    final_answer: str
    route: Literal["auto_answer", "hitl"]
    confidence: float
```

#### 2.5 Graph State Object
```python
class GraphState(TypedDict, total=False):
    query: str
    context_chunks: List[Chunk]
    model_answer: str
    final_answer: str
    route: Literal["answer", "hitl", "end"]
    confidence: float
    needs_escalation: bool
```

---

### 3. Workflow Design (LangGraph)

#### 3.1 Nodes
- **Processing Node (`process_node`)**
  - Input: `GraphState` with at least `query`.
  - Steps:
    1. `context_chunks = retrieve_context(collection, state["query"], k=k)`.
    2. `model_answer = generate_answer(state["query"], context_chunks)`.
    3. `confidence = score_answer(...)`.
    4. `route = route_decision(...)`.
    5. `needs_escalation = (route == "hitl")`.
  - Output: updated `GraphState`.

- **Output Node (`output_node`)**
  - Input: `GraphState` with `route` and `model_answer`.
  - Steps:
    - If `route == "answer"` and `not needs_escalation`:
      - `final_answer = model_answer`.
    - Else:
      - `state = hitl_escalate(state)`.
  - Output: updated `GraphState` with `final_answer`.

#### 3.2 Edges
- `START → process_node`
- `process_node → output_node`
  - Condition inside `output_node` handles routing and escalation.

#### 3.3 State
- Initial state:
  ```python
  {"query": user_query}
  ```
- Final state:
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

### 4. Conditional Routing Logic

#### 4.1 Answer Generation Criteria
- Conditions for auto-answer:
  - At least one relevant chunk retrieved (max similarity score ≥ threshold).
  - Question is in-scope of the PDF topics.
  - Answer does not contain high-risk/blocked keywords.
- Example heuristic:
  ```python
  if chunks and max_score >= SCORE_THRESHOLD and not is_sensitive_intent(query):
      route = "answer"
      needs_escalation = False
  else:
      route = "hitl"
      needs_escalation = True
  ```

#### 4.2 Escalation Criteria
- Any of the following triggers `route = "hitl"`:
  - No chunks or all similarity scores below threshold.
  - Detected sensitive or high-risk intent:
    - Billing disputes, legal issues, medical advice, etc.[web:15][web:21][web:23]
  - LLM expresses uncertainty (e.g., “I’m not sure”, low confidence).
  - User explicitly asks to talk to a human:
    - “agent”, “support representative”, “escalate”, etc.[web:18][web:21]

---

### 5. HITL Design

#### 5.1 When Escalation is Triggered
- During `process_node`, after retrieval and scoring:
  - `needs_escalation = True`
  - `route = "hitl"`

#### 5.2 What Happens After Escalation
- `output_node` detects `route == "hitl"`.
- Calls `hitl_escalate(state)`:
  - Displays query, context, and model answer.
  - Prompts the human agent for action.
- Human provides final text answer or approves the model answer.

#### 5.3 Integration of Human Response
- `hitl_escalate` sets:
  ```python
  state["final_answer"] = human_or_approved_answer
  state["needs_escalation"] = False
  ```
- Control returns to `output_node`, which then prints `final_answer` to CLI.

---

### 6. API / Interface Design

#### 6.1 Input Format
- **Startup Input**
  - String: path to PDF knowledge base (e.g., `/Users/manju/Internships/Project/sample_kb.pdf`).[cite:7][cite:30][cite:32]
- **Runtime Input**
  - Natural language query string from user.

#### 6.2 Output Format
- Text answer displayed in CLI:
  - May include tag:
    - `[AUTO]` for model-only answer.
    - `[HUMAN]` for HITL answer.

#### 6.3 Interaction Flow
1. User runs `python app.py`.
2. System asks for PDF path and indexes it.
3. System prints: “RAG Customer Support Assistant is ready. Type 'exit' to stop.”
4. User enters query.
5. LangGraph executes workflow, returns answer or escalates.
6. Loop until user types `exit`.

---

### 7. Error Handling

#### 7.1 Missing Data
- PDF file not found:
  - Show error and prompt user to re-enter correct path.
- Empty or invalid PDF:
  - Log error, abort indexing, notify user.

#### 7.2 No Relevant Chunks Found
- If `retrieve_context` returns empty list or low scores:
  - Set `needs_escalation = True`.
  - Inform user that the question will be reviewed by a human.

#### 7.3 LLM Failure
- Catch exceptions from LLM API call:
  - Retry a limited number of times with backoff.
  - On repeated failure:
    - Escalate to human.
    - Or return generic error message and log incident.

#### 7.4 Vector Store Errors
- ChromaDB connection or persistence issues:
  - Attempt to re-init store.
  - If error persists, notify user that system is temporarily unavailable.

---

### 8. Summary
The LLD defines concrete modules, data structures, workflow details, conditional routing logic, HITL behavior, interface contracts, and error handling for a production-like RAG Customer Support Assistant built on LangGraph, ChromaDB, and a PDF knowledge base.