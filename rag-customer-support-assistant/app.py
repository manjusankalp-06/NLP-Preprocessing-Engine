import os
from typing import TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# Vector DB configuration
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "support_kb"


class GraphState(TypedDict, total=False):
    """State object that flows through the LangGraph workflow."""
    user_query: str
    retrieved_context: str
    answer: str
    intent: str
    confidence: float
    escalate: bool
    escalation_reason: str
    human_response: str


# -------------------------------
# 1) PDF -> ChromaDB Ingestion
# -------------------------------
def load_pdf_to_vectordb(pdf_path: str):
    """Load a PDF, chunk it, embed it, and store in ChromaDB."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    # Persist to disk
    vectordb.persist()
    return vectordb


def get_vectordb():
    """Return a ChromaDB vector store (existing collection)."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    return vectordb


# -------------------------------
# 2) Intent detection
# -------------------------------
def detect_intent(query: str) -> str:
    q = query.lower()

    if any(word in q for word in ["angry", "complaint", "escalate", "human"]):
        return "escalation"

    if any(word in q for word in ["refund", "cancel", "policy", "price", "feature", "support", "how", "what", "when"]):
        return "faq"

    return "general"

# -------------------------------
# 3) LangGraph nodes
# -------------------------------
def retrieve_node(state: GraphState) -> GraphState:
    """Retrieve relevant context from ChromaDB and decide escalation flag."""
    vectordb = get_vectordb()
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(state["user_query"])
    context = "\n\n".join([doc.page_content for doc in docs])

    intent = detect_intent(state["user_query"])
    # Very simple confidence heuristic
    confidence = 0.85 if context.strip() else 0.25

    escalate = confidence < 0.4 and intent == "escalation"
    reason = ""
    if escalate:
        reason = "Low confidence or escalation intent detected"

    return {
        **state,
        "retrieved_context": context,
        "intent": intent,
        "confidence": confidence,
        "escalate": escalate,
        "escalation_reason": reason,
    }


def answer_node(state: GraphState) -> GraphState:
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",   # ✅ FIXED
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(
        """You are a customer support assistant.
Use only the provided context to answer the user query.
If the answer is not present in the context, clearly say the information is unavailable.

User Query:
{query}

Context:
{context}
"""
    )

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "query": state["user_query"],
        "context": state.get("retrieved_context", ""),
    })

    return {**state, "answer": answer}

def human_escalation_node(state: GraphState) -> GraphState:
    """Human-in-the-loop node for low-confidence or sensitive queries."""
    print("\n--- HUMAN IN THE LOOP REQUIRED ---")
    print("User Query:", state["user_query"])
    print("Reason:", state.get("escalation_reason", "Needs review"))
    print("\nRetrieved context (for reference):")
    print(state.get("retrieved_context", "")[:800], "...\n")  # truncate for CLI

    human_response = input("Enter human support response: ")
    return {
        **state,
        "human_response": human_response,
        "answer": human_response,
        "escalate": False,
    }


def route_after_retrieval(state: GraphState) -> str:
    """Routing function: decide whether to answer or escalate."""
    if state.get("escalate", False):
        return "human_escalation"
    return "answer"


# -------------------------------
# 4) Build LangGraph workflow
# -------------------------------
def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("human_escalation", human_escalation_node)

    workflow.set_entry_point("retrieve")

    # Conditional routing based on state
    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "answer": "answer",
            "human_escalation": "human_escalation",
        },
    )

    workflow.add_edge("answer", END)
    workflow.add_edge("human_escalation", END)

    return workflow.compile()


# -------------------------------
# 5) CLI chat loop
# -------------------------------
def run_chat():
    graph = build_graph()
    print("RAG Customer Support Assistant is ready. Type 'exit' to stop.\n")

    while True:
        user_query = input("User: ").strip()
        if user_query.lower() == "exit":
            break

        result = graph.invoke({"user_query": user_query})

        # Display answer and routing info
        print("\nAssistant:", result.get("answer", "No answer generated"))
        print("Intent:", result.get("intent"))
        print("Confidence:", result.get("confidence"))
        print("Escalated:", "Yes" if result.get("escalate") else "No")
        print("-" * 60)


# -------------------------------
# 6) Main entry point
# -------------------------------
if __name__ == "__main__":
    pdf_path = "sample_kb.pdf"

    import os
    print("File exists:", os.path.exists(pdf_path))  # 👈 ADD THIS LINE

    print("Using PDF path:", pdf_path)

    if not os.path.exists(PERSIST_DIR):
        print("Creating vector database from PDF...")
        load_pdf_to_vectordb(pdf_path)
        print("Vector database created.\n")
    else:
        print(f"Using existing vector database from ./{PERSIST_DIR}\n")

    run_chat()