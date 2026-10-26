from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import qdrant_client


class RAG:
    def __init__(self, config_file, llm):
        self.config = config_file
        self.qdrant_client = qdrant_client.QdrantClient(url=self.config["qdrant_url"])
        self.llm = llm  # ollama llm

    def load_embedder(self):

        embed_model = HuggingFaceEmbedding(model_name=self.config["embedding_model"])
        return embed_model

    def qdrant_index(self):
        client = qdrant_client.QdrantClient(url=self.config["qdrant_url"])
        qdrant_vector_store = QdrantVectorStore(
            client=client, collection_name=self.config["collection_name"]
        )
        Settings.llm = self.llm
        Settings.embed_model = self.load_embedder()
        Settings.chunk_size = self.config["chunk_size"]

        index = VectorStoreIndex.from_vector_store(
            vector_store=qdrant_vector_store, settings=Settings
        )
        return index
