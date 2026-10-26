from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel, Field
import yaml
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from rag.rag import RAG


config_file = "config.yml"

with open(config_file, "r") as conf:
    config = yaml.safe_load(conf)


class Query(BaseModel):
    query: str
    similarity_top_k: Optional[int] = Field(default=1, ge=1, le=5)


class Response(BaseModel):
    search_result: str
    sources: list[str]


llm = Ollama(model=config["llm_name"], base_url=config["llm_url"], request_timeout=300)
rag = RAG(config_file=config, llm=llm)
index = rag.qdrant_index()


app = FastAPI()


@app.get("/")
def root():
    return {"message": "Research RAG"}


qa_prompt = PromptTemplate(
    "Context information from research papers is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "You can only answer based on the provided context. If a response cannot be "
    "formed strictly using the context, politely say you don't have knowledge "
    "about that topic.\n"
    "Query: {query_str}\n"
    "Answer: "
)


@app.post("/api/search", response_model=Response, status_code=200)
def search(query: Query):

    query_engine = index.as_query_engine(
        similarity_top_k=query.similarity_top_k,
        output=Response,
        response_mode="tree_summarize",
        summary_template=qa_prompt,
        verbose=True,
    )
    response = query_engine.query(query.query)
    response_object = Response(
        search_result=str(response).strip(),
        sources=list(
            dict.fromkeys(m["file_path"] for m in (response.metadata or {}).values())
        ),
    )
    return response_object
