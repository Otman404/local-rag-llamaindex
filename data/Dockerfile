FROM rag-base-image

WORKDIR /app

RUN uv sync --frozen --no-install-project --extra data_ingest

COPY data/ .

ENTRYPOINT ["python", "data.py"]
