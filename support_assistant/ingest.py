import os
import chromadb
from chromadb.utils import embedding_functions

def ingest_documents():
    docs_dir = "./docs"
    chroma_path = "./chroma_db"
    
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name="zepto_policies",
        embedding_function=sentence_transformer_ef
    )
    
    documents = []
    metadatas = []
    ids = []
    
    for filename in sorted(os.listdir(docs_dir)):
        if filename.startswith("doc_") and filename.endswith(".txt"):
            doc_id = filename.replace(".txt", "")
            filepath = os.path.join(docs_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            
            documents.append(content)
            metadatas.append({"source": filename})
            ids.append(doc_id)
            
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

if __name__ == "__main__":
    ingest_documents()