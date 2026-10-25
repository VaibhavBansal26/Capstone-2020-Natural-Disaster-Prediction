FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

# Install dependencies first so the layer is cached independently of source changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --all-extras

COPY . .
RUN uv sync --frozen --no-dev --all-extras

# Raw data and trained models live on volumes; the entrypoint fetches and trains what is missing.
# (NLTK corpora are downloaded by the app at startup.)
VOLUME ["/app/data/raw", "/app/models"]
EXPOSE 8000
ENTRYPOINT ["./deploy/entrypoint.sh"]
