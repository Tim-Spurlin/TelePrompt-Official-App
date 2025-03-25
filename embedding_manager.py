############################################################
# embedding_manager.py
# Handles building, saving, and loading a FAISS vector index
############################################################

import os
import logging
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import normalize
import json
import threading
from kivy.clock import Clock  # for callbacks on the main thread if needed
import datetime

logger = logging.getLogger(__name__)

def chunk_text(text, chunk_size=50, overlap=300):
    """
    Splits the given text into chunks with an overlap.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

class EmbeddingManager:
    _instance = None  # Singleton instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, model_name="local_models/multi-qa-mpnet-base-dot-v1", top_k=5):
        if hasattr(self, "__initialized") and self.__initialized:
            return  # Prevent re-initialization
        logger.info(f"Loading embedding model: {model_name}")
        # Load the model solely from the local files.
        self.model = SentenceTransformer(model_name, local_files_only=True)
        self.top_k = top_k  # Number of top results to return
        self.index = None
        self.doc_map = {}
        # Dictionaries to hold per-preset conversation indexes and doc_maps
        self.conversation_indexes = {}   # key: preset folder path, value: FAISS index for conversation history
        self.conversation_doc_maps = {}    # key: preset folder path, value: dict mapping index -> (utterance, timestamp)
        # In-memory recent history for immediate recall per preset.
        self.recent_history_per_preset = {}  # Key: preset folder -> list of (utterance, timestamp)
        self.__initialized = True

    def build_index(self, doc_texts, index_path, doc_map_path):
        all_chunks = []
        self.doc_map = {}  # Mapping: key = chunk index, value = (doc_id, chunk_text)
        current_index = 0

        for doc_id, text_content in doc_texts:
            chunks = chunk_text(text_content, chunk_size=250, overlap=100)
            for ch in chunks:
                self.doc_map[current_index] = (doc_id, ch)
                all_chunks.append(ch)
                current_index += 1

        logger.info(f"Embedding {len(all_chunks)} chunks...")
        embeddings = self.model.encode(all_chunks, convert_to_tensor=False, show_progress_bar=True)
        embeddings = normalize(embeddings, norm="l2")
        dim = embeddings.shape[1]
        num_chunks = len(embeddings)

        logger.info("Building HNSW index...")
        index = faiss.IndexHNSWFlat(dim, 32)  # HNSW with M=32
        index.hnsw.efSearch = 500             # Set efSearch for high recall
        index.add(embeddings)
        self.index = index
        logger.info(f"HNSW index built with {self.index.ntotal} vectors.")

        faiss.write_index(self.index, index_path)
        logger.info(f"FAISS index saved at: {index_path}")

        np.save(doc_map_path, self.doc_map, allow_pickle=True)
        logger.info(f"Doc map saved at: {doc_map_path} with {len(self.doc_map)} entries.")

    def load_index(self, index_path, doc_map_path):
        logger.info(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)
        logger.info(f"Loading doc map from {doc_map_path}")
        loaded = np.load(doc_map_path, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.ndim == 0:
            self.doc_map = loaded.item()
        else:
            self.doc_map = loaded
        logger.info(f"Doc map loaded, type: {type(self.doc_map)}, length: {len(self.doc_map)}")
        if hasattr(self.index, "ntotal"):
            assert self.index.ntotal == len(self.doc_map), "FAISS index and doc_map are misaligned!"

    def query(self, query_text, top_k=None):
        if not self.index:
            raise ValueError("FAISS index is not loaded. Call load_index() first.")
        if top_k is None:
            top_k = self.top_k
        query_emb = self.model.encode([query_text], convert_to_tensor=False)
        query_emb = np.array(query_emb, dtype="float32")
        distances, indices = self.index.search(query_emb, top_k)
        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            doc_id, chunk_text = self.doc_map[idx]
            score = distances[0][rank]
            results.append((doc_id, chunk_text, score))
        return results

    def add_document(self, text, file_path):
        vector = self.model.encode([text])[0]
        vec = np.array([vector], dtype="float32")
        if vec.ndim != 2:
            vec = vec.reshape(1, -1)
        if self.index is None:
            d = vec.shape[1]
            self.index = faiss.IndexFlatL2(d)
        self.index.add(vec)
        doc_idx = self.index.ntotal - 1
        self.doc_map[doc_idx] = file_path

    def save_index(self, index_path, doc_map_path):
        faiss.write_index(self.index, index_path)
        np.save(doc_map_path, self.doc_map)
        print(f"Saved FAISS index to {index_path} and doc map to {doc_map_path}")

    def add_documents(self, new_doc_texts, index_path, doc_map_path):
        import os
        if os.path.exists(index_path) and os.path.exists(doc_map_path):
            self.load_index(index_path, doc_map_path)
        else:
            self.index = None
            self.doc_map = {}

        all_new_chunks = []
        new_chunk_map = {}
        start_index = self.index.ntotal if self.index is not None else 0
        current_index = start_index

        for doc_id, text_content in new_doc_texts:
            chunks = chunk_text(text_content, chunk_size=250, overlap=100)
            for ch in chunks:
                new_chunk_map[current_index] = (doc_id, ch)
                all_new_chunks.append(ch)
                current_index += 1

        if not all_new_chunks:
            print("No new chunks found, skipping update.")
            return

        new_embeddings = self.model.encode(all_new_chunks, convert_to_tensor=False, show_progress_bar=True)
        new_embeddings = normalize(new_embeddings, norm="l2")
        dim = new_embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexHNSWFlat(dim, 32)
            self.index.hnsw.efSearch = 500

        self.index.add(new_embeddings)
        self.doc_map.update(new_chunk_map)
        faiss.write_index(self.index, index_path)
        np.save(doc_map_path, self.doc_map, allow_pickle=True)
        print("FAISS index and doc_map updated with new documents.")

    def async_update_index(self, new_doc_texts, index_path, doc_map_path, on_update_done=None):
        def background_task():
            import os
            if os.path.exists(index_path) and os.path.exists(doc_map_path):
                self.load_index(index_path, doc_map_path)
            else:
                self.index = None
                self.doc_map = {}

            all_new_chunks = []
            new_chunk_map = {}
            start_index = self.index.ntotal if self.index is not None else 0
            current_index = start_index
            for doc_id, text_content in new_doc_texts:
                chunks = chunk_text(text_content, chunk_size=250, overlap=100)
                for ch in chunks:
                    new_chunk_map[current_index] = (doc_id, ch)
                    all_new_chunks.append(ch)
                    current_index += 1

            if not all_new_chunks:
                print("No new chunks found, skipping update.")
                return

            new_embeddings = self.model.encode(all_new_chunks, convert_to_tensor=False, show_progress_bar=True)
            new_embeddings = normalize(new_embeddings, norm="l2")
            dim = new_embeddings.shape[1]

            if self.index is None:
                self.index = faiss.IndexHNSWFlat(dim, 32)
                self.index.hnsw.efSearch = 500

            self.index.add(new_embeddings)
            self.doc_map.update(new_chunk_map)
            faiss.write_index(self.index, index_path)
            np.save(doc_map_path, self.doc_map, allow_pickle=True)
            print("FAISS index and doc_map updated with new documents.")

            if on_update_done is not None:
                Clock.schedule_once(lambda dt: on_update_done(), 0)

        threading.Thread(target=background_task, daemon=True).start()

    # ---------------- Conversation History Methods ----------------

    def load_conversation_index(self, preset_folder: str):
        """
        Loads (or initializes) the conversation index and document map for the given preset.
        These are stored as binary files in the preset folder.
        """
        conv_index_path = os.path.join(preset_folder, "conversation_index.faiss")
        embedding_managerp_path = os.path.join(preset_folder, "conversation_doc_map.npy")
        vector_dim = self.model.encode(["test"])[0].shape[0]
        try:
            conv_index = faiss.read_index(conv_index_path)
            conv_doc_map = np.load(embedding_managerp_path, allow_pickle=True).item()
            logger.info(f"Loaded conversation index from {conv_index_path}")
        except Exception as e:
            logger.info(f"No existing conversation index found in {preset_folder}, initializing new one. Reason: {e}")
            conv_index = faiss.IndexHNSWFlat(vector_dim, 32)
            conv_index.hnsw.efSearch = 500
            conv_doc_map = {}
        self.conversation_indexes[preset_folder] = conv_index
        self.conversation_doc_maps[preset_folder] = conv_doc_map
        self.recent_history_per_preset.setdefault(preset_folder, [])

    def save_conversation_index(self, preset_folder: str):
        """
        Saves the conversation index and document map for the given preset.
        """
        conv_index_path = os.path.join(preset_folder, "conversation_index.faiss")
        embedding_managerp_path = os.path.join(preset_folder, "conversation_doc_map.npy")
        faiss.write_index(self.conversation_indexes[preset_folder], conv_index_path)
        np.save(embedding_managerp_path, self.conversation_doc_maps[preset_folder], allow_pickle=True)
        logger.info(f"Saved conversation index to {conv_index_path} and doc map to {embedding_managerp_path}")

    def add_conversation_utterance(self, utterance: str, preset_folder: str, who: str):
        """
        Immediately appends a new conversation utterance (with timestamp and role) to the conversation index
        and document map for the given preset.
        """
        if preset_folder not in self.conversation_indexes:
            self.load_conversation_index(preset_folder)
        # Encode the utterance and add it to the FAISS index.
        vector = self.model.encode([utterance], convert_to_tensor=False)
        vector = np.array(vector, dtype="float32").reshape(1, -1)
        conv_index = self.conversation_indexes[preset_folder]
        conv_index.add(vector)
        new_index = conv_index.ntotal - 1
        timestamp = datetime.datetime.now().isoformat()
        entry = {"who": who, "content": utterance, "timestamp": timestamp}
        self.conversation_doc_maps[preset_folder][new_index] = entry
        # Update the in‑memory recent history list.
        self.recent_history_per_preset.setdefault(preset_folder, []).append(entry)
        self.save_conversation_index(preset_folder)
        logger.info(f"Appended {who} utterance in preset '{preset_folder}' at index {new_index} with timestamp {timestamp}")

    def query_conversation_history(self, query_text: str, preset_folder: str, top_k: int = 5):
        """
        Performs semantic search on the entire conversation history for the given preset.
        Returns a list of dictionaries with keys "who" and "content".
        """
        if preset_folder not in self.conversation_indexes:
            self.load_conversation_index(preset_folder)
        conv_index = self.conversation_indexes[preset_folder]
        vector = self.model.encode([query_text], convert_to_tensor=False)
        vector = np.array(vector, dtype="float32")
        distances, indices = conv_index.search(vector, top_k)
        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            if idx in self.conversation_doc_maps[preset_folder]:
                entry = self.conversation_doc_maps[preset_folder][idx]
                results.append(entry)
        return results

    def get_all_messages(self, preset_folder):
        """
        Returns all conversation entries in chronological order
        from the doc_map for the given preset.
        Each entry is the same type you store in add_conversation_utterance
        (e.g., tuple or dict).
        """
        if preset_folder not in self.conversation_doc_maps:
            self.load_conversation_index(preset_folder)
        doc_map = self.conversation_doc_maps[preset_folder]

        # doc_map keys are index integers, so sort them.
        sorted_keys = sorted(doc_map.keys())
        messages = [doc_map[k] for k in sorted_keys]
        return messages

    def get_last_n_messages(self, preset_folder, n=5):
        """
        Returns only the last N messages in chronological order.
        """
        all_msgs = self.get_all_messages(preset_folder)
        return all_msgs[-n:]  # The last N messages

# Global singleton instance
embedding_manager = EmbeddingManager()
# The embedder is already available as embedding_manager.model.
# If needed, you can reference it as:
# my_embedder = embedding_manager.model

def verify_faiss_alignment():
    """
    Checks that the FAISS index and doc_map.npy contain the same number of items.
    """
    index_size = embedding_manager.index.ntotal if embedding_manager.index is not None else 0
    doc_map_size = len(embedding_manager.doc_map)
    assert index_size == doc_map_size, f"FAISS index ({index_size}) and doc_map.npy ({doc_map_size}) are misaligned!"
    print(f"FAISS index and document map are correctly aligned: {index_size} entries.")
