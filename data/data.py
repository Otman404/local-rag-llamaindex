from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
)
import qdrant_client
import argparse
import arxiv
import os
import structlog
from dotenv import load_dotenv

logger = structlog.get_logger()

load_dotenv()

data_path = os.getenv("DATA_PATH")
qdrant_url = os.getenv("QDRANT_URL")
collection_name = os.getenv("COLLECTION_NAME")
embedding_model = os.getenv("EMBEDDING_MODEL")
llm_url = os.getenv("LLM_URL")
llm_name = os.getenv("LLM_NAME")
chunk_size = int(os.getenv("CHUNK_SIZE"))


class Data:
    def __init__(self):
        pass

    def _create_data_folder(self, download_path):
        data_path = download_path
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            logger.info("Output folder created", output_folder=data_path)
        else:
            logger.warning(f"Output folder already exists at {data_path}")

    def download_papers(self, search_query, download_path, max_results):
        self._create_data_folder(download_path)
        client = arxiv.Client()
        logger.info("Searching papers from arxiv...", query=search_query)
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        results = list(client.results(search))
        for i, paper in enumerate(results):
            if os.path.exists(download_path):
                paper_title = (paper.title).replace(" ", "_")
                paper.download_pdf(dirpath=download_path, filename=f"{paper_title}.pdf")
                logger.info(
                    f"{i+1}/{len(results)} PDF Downloaded", pdf_name=paper.title
                )

    def ingest(self, embedder, llm):
        logger.info("Indexing data...")
        documents = SimpleDirectoryReader(data_path).load_data()

        client = qdrant_client.QdrantClient(url=qdrant_url)
        qdrant_vector_store = QdrantVectorStore(
            client=client, collection_name=collection_name
        )
        storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)

        Settings.llm = None
        Settings.embed_model = embedder
        Settings.chunk_size = chunk_size
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, Settings=Settings
        )
        logger.info(
            "Data indexed successfully to Qdrant",
            collection=collection_name,
        )
        return index


def main():
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
    data = Data()
    if args.query:
        data.download_papers(
            search_query=args.query,
            download_path=data_path,
            max_results=args.max,
        )
    if args.ingest:
        logger.info("Loading Embedder...", embed_model=embedding_model)
        embed_model = HuggingFaceEmbedding(
            model_name=embedding_model, cache_folder="./cache"
        )
        llm = Ollama(model=llm_name, base_url=llm_url)
        data.ingest(embedder=embed_model, llm=llm)


if __name__ == "__main__":
    main()
