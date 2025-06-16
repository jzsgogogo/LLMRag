
from chromadb import PersistentClient
from chromadb.config import Settings

from Model_loader import CHROMA_PERSIST_DIR


def init_chroma_client():
    client = PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    try:
        collection = client.get_collection("enterprise_docs")
        collection.clear()
    except Exception:
        pass
    return client

import uuid

def add_to_chroma(client, docs_split, vectors):
    collection = client.get_or_create_collection(
        name="enterprise_docs",
        metadata={"hnsw:space": "cosine"}
    )
    unique_ids = [str(uuid.uuid4()) for _ in docs_split]
    collection.add(
        ids=unique_ids,
        documents=[d.page_content for d in docs_split],
        metadatas=[d.metadata for d in docs_split],
        embeddings=vectors
    )