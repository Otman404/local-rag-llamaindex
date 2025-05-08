FROM rag-base-image

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv pip install --system ".[data_ingest]"

COPY data/ .

ENTRYPOINT ["uv", "run", "data.py"]
