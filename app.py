from fastapi import FastAPI
from typing import Optional, List
import ollama
from pydantic import BaseModel, Field
import yaml
from llama_index.llms import Ollama

from rag.rag import RAG


config_file = "config.yml"

with open(config_file, "r") as conf:
    config = yaml.safe_load(conf)


class Query(BaseModel):
    query: str
    similarity_top_k: Optional[int] = Field(default=1, ge=1, le=10)


class Response(BaseModel):
    search_result: str 
    source: str


llm = Ollama(model="zeph", url=config["llm_url"])
rag = RAG(config_file=config, llm=llm)
index = rag.qdrant_index()


app = FastAPI()


@app.get("/")
def root():
    return {"message": "Research RAG"}


@app.post("/api/search", response_model=Response, status_code=200)
def search(query: Query):
    query_engine = index.as_query_engine(similarity_top_k=query.similarity_top_k, output=Response)
    response = query_engine.query(query.query)
    response_object = Response(
        search_result=str(response).strip(), source=[response.metadata[k]["file_path"] for k in response.metadata.keys()][0]
    )
    return response_object