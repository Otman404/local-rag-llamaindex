from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from tqdm import tqdm
import arxiv
import os
import argparse
import yaml
import qdrant_client
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding

from llama_index import ServiceContext
from llama_index.llms import Ollama


class Data:
    def __init__(self, config):
        self.config = config

    def _create_data_folder(self, download_path):
        data_path = download_path
        if not os.path.exists(data_path):
            os.makedirs(self.data_folder)
            print("Output folder created")
        else:
            print("Output folder already exists.")

    def download_papers(self, search_query, download_path, max_results):
        self._create_data_folder(download_path)
        client = arxiv.Client()

        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        results = list(client.results(search))
        for paper in tqdm(results):
            if os.path.exists(download_path):
                paper_title = (paper.title).replace(" ", "_")
                paper.download_pdf(dirpath=download_path, filename=f"{paper_title}.pdf")
                print(f"{paper.title} Downloaded.")

    def ingest(self, embedder, llm):
        print("Indexing data...")
        documents = SimpleDirectoryReader(self.config["data_path"]).load_data()

        client = qdrant_client.QdrantClient(url=self.config["qdrant_url"])
        qdrant_vector_store = QdrantVectorStore(
            client=client, collection_name=self.config["collection_name"]
        )
        storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)
        # service_context = ServiceContext.from_defaults(
        #     llm=llm, embed_model=embedder, chunk_size=self.config["chunk_size"]
        # )
        service_context = ServiceContext.from_defaults(
            llm=None, embed_model=embedder, chunk_size=self.config["chunk_size"]
        )

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, service_context=service_context
        )
        print(f"Data indexed successfully to Qdrant. Collection: {self.config['collection_name']}")
        return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        default=False,
        help="Download papers from arxiv with this query.",
    )
    parser.add_argument("--output", type=str, default=False, help="Download path.")

    parser.add_argument(
        "--max", type=int, default=False, help="Max results to download."
    )

    parser.add_argument(
        "--ingest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ingest data to qdrant vector Database.",
    )

    args = parser.parse_args()
    config_file = "config.yml"
    with open(config_file, "r") as conf:
        config = yaml.safe_load(conf)
    data = Data(config)
    if args.query:
        data.download_papers(
            search_query=args.query, download_path=args.output, max_results=args.max
        )
    if args.ingest:
        print("Loading Embedder...")
        embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name=config["embedding_model"])
        )
        llm = Ollama(model="zeph", base_url=config["llm_url"])
        data.ingest(embedder=embed_model, llm=llm)
