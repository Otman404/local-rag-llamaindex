FROM rag-base-image

WORKDIR /api

COPY pyproject.toml uv.lock ./

RUN uv pip install --system ".[api]"

COPY api/ .

CMD ["fastapi", "run", "main.py", "--port", "80", "--host", "0.0.0.0"]
