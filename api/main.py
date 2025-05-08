from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel, Field
from llama_index.llms.ollama import Ollama
from rag import RAG
from dotenv import load_dotenv
import os

load_dotenv()


llm_url = os.getenv("LLM_URL")
llm_name = os.getenv("LLM_NAME")


class Query(BaseModel):
    query: str
    similarity_top_k: Optional[int] = Field(default=1, ge=1, le=5)


class Response(BaseModel):
    search_result: str
    source: str


llm = Ollama(model=llm_name, base_url=llm_url)
rag = RAG(llm=llm)
index = rag.qdrant_index()


app = FastAPI()


@app.get("/")
def root():
    return {"message": "Research RAG"}


a = "You can only answer based on the provided context. If a response cannot be formed strictly using the context, politely say you don’t have knowledge about that topic"


@app.post("/api/search", response_model=Response, status_code=200)
def search(query: Query):

    query_engine = index.as_query_engine(
        similarity_top_k=query.similarity_top_k,
        output=Response,
        response_mode="tree_summarize",
        verbose=True,
    )
    response = query_engine.query(query.query + a)
    response_object = Response(
        search_result=str(response).strip(),
        source=[response.metadata[k]["file_path"] for k in response.metadata.keys()][0],
    )
    return response_object
