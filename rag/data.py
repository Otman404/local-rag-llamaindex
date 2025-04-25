from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
)
from tqdm import tqdm
import qdrant_client
import argparse
import arxiv
import yaml
import os
import structlog

logger = structlog.get_logger()


class Data:
    def __init__(self, config):
        self.config = config

    def _create_data_folder(self, download_path):
        data_path = download_path
        if not os.path.exists(data_path):
            os.makedirs(self.config["data_path"])
            logger.info("Output folder created", output_folder=data_path)
        else:
            logger.warning(f"Output folder already exists at {data_path}")

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
                logger.info("PDF Downloaded", pdf_name=paper.title)

    def ingest(self, embedder, llm):
        logger.info("Indexing data...")
        documents = SimpleDirectoryReader(self.config["data_path"]).load_data()

        client = qdrant_client.QdrantClient(url=self.config["qdrant_url"])
        qdrant_vector_store = QdrantVectorStore(
            client=client, collection_name=self.config["collection_name"]
        )
        storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)

        Settings.llm = None
        Settings.embed_model = embedder
        Settings.chunk_size = self.config["chunk_size"]
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, Settings=Settings
        )
        logger.info(
            "Data indexed successfully to Qdrant",
            collection=self.config["collection_name"],
        )
        return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default=False,
        help="Download papers from arxiv with this query.",
    )
    # parser.add_argument(
    #     "-o", "--output", type=str, default=False, help="Download path."
    # )

    parser.add_argument(
        "-m", "--max", type=int, default=False, help="Max results to download."
    )

    parser.add_argument(
        "-i",
        "--ingest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ingest data to Qdrant vector Database.",
    )

    args = parser.parse_args()
    config_file = "config.yml"
    with open(config_file, "r") as conf:
        config = yaml.safe_load(conf)
    data = Data(config)
    if args.query:
        data.download_papers(
            search_query=args.query,
            download_path=config["data_path"],
            max_results=args.max,
        )
    if args.ingest:
        logger.info("Loading Embedder...", embed_model=config["embedding_model"])
        embed_model = HuggingFaceEmbedding(model_name=config["embedding_model"])
        llm = Ollama(model=config["llm_name"], base_url=config["llm_url"])
        data.ingest(embedder=embed_model, llm=llm)
