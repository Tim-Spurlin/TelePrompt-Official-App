# semantic_search.py

from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load the local model for real-time encoding
model = SentenceTransformer("local_models/multi-qa-mpnet-base-dot-v1", local_files_only=True)

def embed_and_compare(query, docs):
    """
    Given a query string and a list of document strings (docs),
    returns the document text that is most similar to the query.
    """
    # Compute embeddings for the query and documents
    query_embedding = model.encode(query)
    doc_embeddings = model.encode(docs)

    # Use util.cos_sim() for efficient cosine similarity calculation.
    similarities = util.cos_sim(query_embedding, doc_embeddings)
    
    best_index = int(np.argmax(similarities))
    return docs[best_index]

def retrieve_conversation_history(query_text, top_k=5):
    """
    Searches FAISS for the most relevant past conversations related to the query.
    """
    from embedding_manager import embedding_manager

    # Query FAISS for relevant past interactions
    results = embedding_manager.query(query_text, top_k)
    
    history = []
    for doc_id, chunk_text, score in results:
        # Return each item as a dictionary so that create_prompt() can access message["who"] and message["content"]
        history.append({"who": "assistant", "content": chunk_text})
    
    return history  # Corrected to return history, not results