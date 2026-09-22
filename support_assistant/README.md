Module 3 - Support Assistant (RAG Pipeline)
How It Works

The GenAI Customer Support Assistant uses a Retrieval-Augmented Generation (RAG) pipeline to provide answers based on the available company policies. The process is divided into four main stages:

Ingestion: ingest.py reads the 8 policy documents, extracts their content, and breaks the information into smaller chunks for processing.
Embedding: Each policy chunk is converted into a numerical representation using the all-MiniLM-L6-v2 model from sentence-transformers. The embeddings are generated locally.
Retrieval: The generated embeddings are stored in a persistent ChromaDB database (./chroma_db). When a user asks a question, the system searches the database and retrieves the top 3 most relevant policy chunks using cosine similarity.
Generation & Orchestration: LangGraph manages the flow of the application, deciding whether the question can be answered using the available policies or whether a general fallback response is needed. The final response is also validated using a Pydantic schema to ensure a consistent output format.
Example API Call

The following example shows how a user can ask the support assistant a policy-related question.

Example 1: Policy Retrieval Question

POST /ask

{
  "query": "What is the delivery fee for orders under 149?"
}

The system searches the policy database for the relevant delivery-fee information and generates a response based on the retrieved policy content.