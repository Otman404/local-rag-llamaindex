from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import qdrant_client
from dotenv import load_dotenv
import os

load_dotenv()


qdrant_url = os.getenv("QDRANT_URL")
collection_name = os.getenv("COLLECTION_NAME")
embedding_model = os.getenv("EMBEDDING_MODEL")
chunk_size = int(os.getenv("CHUNK_SIZE"))


class RAG:
    def __init__(self, llm):
        self.qdrant_client = qdrant_client.QdrantClient(url=qdrant_url)
        self.llm = llm  # ollama llm

    def load_embedder(self):

        embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        return embed_model

    def qdrant_index(self):
        client = qdrant_client.QdrantClient(url=qdrant_url)
        qdrant_vector_store = QdrantVectorStore(
            client=client, collection_name=collection_name
        )
        Settings.llm = self.llm
        Settings.embed_model = self.load_embedder()
        Settings.chunk_size = chunk_size

        index = VectorStoreIndex.from_vector_store(
            vector_store=qdrant_vector_store, settings=Settings
        )
        return index
