FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:debian

WORKDIR /app

COPY pyproject.toml /app/
## TODO: Check out --compile-bytecode
## Install dependencies inside of image (from the pyproject.toml-file)
RUN uv sync --no-install-project --no-cache
ENV PATH="/app/.venv/bin:$PATH"

## docker build --no-cache -t halyjo/test-image:latest . --progress=plain
## docker push halyjo/test-image:latest
