FROM rag-base-image

WORKDIR /app

RUN uv sync --frozen --no-install-project --extra api

WORKDIR /api

COPY api/ .

CMD ["fastapi", "run", "main.py", "--port", "80", "--host", "0.0.0.0"]
