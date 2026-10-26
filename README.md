# RAG: Research-assistant

![Header](images/readme_header.png)

This project aims to help researchers find answers from a set of research papers with the help of a customized RAG pipeline and a powerfull LLM, all offline and free of cost.

For more details, please checkout the [blog post](https://otmaneboughaba.com/posts/local-rag-api) about this project.

## How it works

![Project Architecture](images/local-rag-architecture.png)

1. Download some research papers from Arxiv
2. Use Llamaindex to load, chunk, embed and store these documents to a Qdrant database
3. FastAPI endpoint that receives a query/question, searches through our documents and find the best matching chunks
4. Feed these relevant documents into an LLM as a context
5. Generate an easy to understand answer and return it as an API response alongside citing the sources

## Running the project

### Quick Start (Docker Compose)

This project now uses Docker and Docker Compose for easier setup and reproducibility: 
_You can visit this blog article to learn more: [Dockerizing a RAG Application with FastAPI, LlamaIndex, Qdrant and Ollama](https://otmaneboughaba.com/posts/dockerize-rag-application/)_

#### 1. Clone the Repo

```bash
git clone https://github.com/Otman404/local-rag-llamaindex
cd local-rag-llamaindex
```

#### 2. Download the LLM

The Ollama container builds the model from `ollama/Modelfile`, which expects the [Zephyr 7B Beta](https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF) GGUF file (~5 GB) inside the `ollama/` folder:

```bash
curl -L -o ollama/zephyr-7b-beta.Q5_K_M.gguf \
  https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q5_K_M.gguf
```

#### 3. Build the Docker images

```bash
docker compose --profile data-ingest build
```

This will:

* Build a rag-base-image containing the shared dependencies (LlamaIndex, Qdrant client, sentence-transformers, etc.)

* Build the api service (FastAPI backend) using Dockerfile in api/

* Build the data_ingestion service used for downloading and indexing research papers using Dockerfile in data/ (it sits behind the `data-ingest` profile, hence the `--profile` flag)

#### 4. Start Qdrant and Ollama

```bash
docker compose up -d qdrant ollama
```

This starts:

* **Qdrant** vector DB (port `6333`)
* **Ollama** LLM server (port `11434`)

#### 5. Download & Ingest Papers

To ingest new research papers into Qdrant:

```bash
docker compose run --rm data_ingestion --query "LLM" --max 5 --ingest
```

`--query`: search keyword for downloading research papers from Arxiv.

`--max`: Limits the results.

`--ingest`: Ingests the downloaded papers into the Qdrant database.

This uses `arxiv` to fetch papers, chunk them with LlamaIndex, and store them into Qdrant.

#### 6. Start the API

```bash
docker compose up api
```

The API server runs on port `8000`.

> **Note:** ingest at least once before starting the API. The API detects the Qdrant collection's format when it starts, so if it starts before the collection exists, queries fail with `Not existing vector name error: text-dense`. If that happens, run `docker compose restart api`. Ingesting more papers later doesn't need a restart.

#### 7. Query the API

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main findings?", "similarity_top_k": 2}'
```

Or use the interactive docs at `http://localhost:8000/docs`.

The response contains the generated answer and the list of papers it was based on:

```json
{
  "search_result": "...",
  "sources": ["/app/papers/Paper_A.pdf", "/app/papers/Paper_B.pdf"]
}
```

> Running on CPU (e.g. Docker on a Mac), a query can take a minute or more.

#### Stopping & Clean Up

To stop all running containers and networks started by Docker Compose:

```bash
docker compose down
```

### Running the project manually with uv

#### Starting a Qdrant docker instance

```bash
docker run -p 6333:6333 -v ~/qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

#### Setting up the environment

Inside the project folder run

```bash
# create virtual env with uv
uv venv
# activate the virtual env
source .venv/bin/activate
# install project dependencies
uv pip install ".[dev]"
```

#### Downloading & Indexing data

```bash
uv run data/data.py --query "LLM" --max 5 --ingest
```

#### Starting Ollama LLM server

Follow [this article](https://otmaneboughaba.com/posts/local-llm-ollama-huggingface/) for more infos on how to run models from hugging face locally with Ollama.

Create model from Modelfile

```bash
ollama create research_assistant -f ollama/Modelfile 
```

Start the model server

```bash
ollama run research_assistant
```

By default, Ollama runs on ```http://localhost:11434```

#### Starting the api server

```bash
uv run fastapi api/main.py dev
```

## Example

### Request

![Post Request](images/post_request.png)

### Response

![Response](images/response.png)
