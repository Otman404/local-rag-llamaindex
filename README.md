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

#### Starting a Qdrant docker instance

```bash
docker run -p 6333:6333 -v ~/qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

#### Setting up the environment

```bash
# clone the repository
git clone https://github.com/Otman404/local-rag-llamaindex
cd local-rag-llamaindex

# create virtual env with uv
uv venv

# activate the virtual env
source .venv/bin/activate
```

#### Downloading & Indexing data

```bash
uv run rag/data.py --query "LLM" --max 10 --ingest
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
uv run fastapi dev
```


## Example

#### Request

![Post Request](images/post_request.png)

#### Response
![Response](images/response.png)
