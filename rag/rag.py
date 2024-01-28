### Loading the embedder

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext
from llama_index import ServiceContext
from llama_index import VectorStoreIndex
from llama_index import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import qdrant_client
import yaml

class RAG:
    def __init__(self, config_file, llm):
        self.config = config_file
        self.qdrant_client = qdrant_client.QdrantClient(
            url=self.config['qdrant_url']
        )
        self.llm = llm  # ollama llm
    
    def load_embedder(self):
        embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name=self.config['embedding_model'])
        )
        return embed_model


    def qdrant_index(self):
        client = qdrant_client.QdrantClient(url=self.config["qdrant_url"])
        qdrant_vector_store = QdrantVectorStore(
            client=client, collection_name=self.config['collection_name']
        )
        # service_context = ServiceContext.from_defaults(
        #     llm=self.llm, embed_model="local:BAAI/bge-small-en-v1.5"
        # )

        service_context = ServiceContext.from_defaults(
            llm=self.llm, embed_model=self.load_embedder(), chunk_size=self.config["chunk_size"]
        )

        index = VectorStoreIndex.from_vector_store(
            vector_store=qdrant_vector_store, service_context=service_context
        )
        return index

    # def _load_config(self, config_file):
    #     with open(config_file, "r") as stream:
    #         try:
    #             self.config = yaml.safe_load(stream)
    #         except yaml.YAMLError as e:
    #             print(e)

