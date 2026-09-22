import os
from typing import List, TypedDict
from fastapi import FastAPI
from pydantic import BaseModel, Field
import chromadb
from chromadb.utils import embedding_functions
from langgraph.graph import StateGraph, END

STRUCTURED_PROMPT_TEMPLATE = """
Role: You are an official Zepto Customer Support AI assistant.
Context: {context}
Task: Answer the user's query accurately using only the provided policy context.
Format: Return a JSON object with keys "answer", "sources", and "confidence".
Length: Concise, maximum 3 sentences for the answer.

Constraints:
- Do not answer using information not present in the provided context.
- If the context does not contain enough information, state clearly that information is unavailable.

Few-Shot Example:
Context: "Zepto delivers grocery and household essentials within 10 to 30 minutes. Standard delivery is free on orders over INR 149."
Query: "What is the minimum order for free delivery?"
Response: {{
    "answer": "Standard delivery is free on orders over INR 149.",
    "sources": ["doc_01"],
    "confidence": 1.0
}}

User Query: {query}
"""

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="zepto_policies",
    embedding_function=sentence_transformer_ef
)

class GraphState(TypedDict):
    query: str
    intent: str
    context: List[str]
    sources: List[str]
    answer: str
    confidence: float

def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]
    mock_env = os.getenv("MOCK_LLM", "1")
    
    if mock_env != "0":
        keywords = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]
        query_lower = query.lower()
        if any(kw in query_lower for kw in keywords):
            intent = "policy_question"
        else:
            intent = "general_question"
    else:
        keywords = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]
        intent = "policy_question" if any(kw in query.lower() for kw in keywords) else "general_question"
        
    return {**state, "intent": intent}

def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]
    results = collection.query(query_texts=[query], n_results=3)
    
    docs = results["documents"][0] if results and results.get("documents") else []
    ids = results["ids"][0] if results and results.get("ids") else []
    
    mock_env = os.getenv("MOCK_LLM", "1")
    
    if mock_env != "0":
        top_snippet = docs[0][:200] if docs else ""
        answer = f"Based on the retrieved context: {top_snippet}"
        sources = ids
        confidence = 1.0
    else:
        top_snippet = docs[0][:200] if docs else ""
        answer = f"Based on the retrieved context: {top_snippet}"
        sources = ids
        confidence = 1.0
        
    return {**state, "context": docs, "sources": sources, "answer": answer, "confidence": confidence}

def direct_answer(state: GraphState) -> GraphState:
    mock_env = os.getenv("MOCK_LLM", "1")
    
    if mock_env != "0":
        answer = "I can only answer questions about Zepto policies right now."
    else:
        answer = "I can only answer questions about Zepto policies right now."
        
    return {**state, "sources": [], "answer": answer, "confidence": 1.0}

def route_intent(state: GraphState) -> str:
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"

workflow = StateGraph(GraphState)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.set_entry_point("classify_intent")
workflow.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)
workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

app_graph = workflow.compile()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float = Field(ge=0.0, le=1.0)

app = FastAPI(title="Zepto Customer Support Assistant")

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    initial_state = {
        "query": request.query,
        "intent": "",
        "context": [],
        "sources": [],
        "answer": "",
        "confidence": 0.0
    }
    
    final_state = app_graph.invoke(initial_state)
    
    return QueryResponse(
        answer=final_state["answer"],
        sources=final_state["sources"],
        confidence=final_state["confidence"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)