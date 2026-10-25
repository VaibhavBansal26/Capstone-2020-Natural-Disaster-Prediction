FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

# Install dependencies first so the layer is cached independently of source changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --all-extras

COPY . .
RUN uv sync --frozen --no-dev --all-extras

# Raw data, trained models and NLTK corpora live on volumes; the entrypoint fills them.
VOLUME ["/app/data/raw", "/app/models"]
EXPOSE 8000
ENTRYPOINT ["./deploy/entrypoint.sh"]
