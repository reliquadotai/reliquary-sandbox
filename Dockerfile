ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3

FROM ${PYTHON_IMAGE} AS guest
RUN test -x /bin/bash \
    && groupadd --gid 1000 sandbox \
    && useradd --uid 1000 --gid 1000 --home-dir /workspace --no-create-home sandbox \
    && mkdir /workspace \
    && chown 1000:1000 /workspace
WORKDIR /workspace
USER 1000:1000
CMD ["python", "-c", "import time; time.sleep(2147483647)"]

FROM ${PYTHON_IMAGE} AS controller-build
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
WORKDIR /app
RUN pip install --no-cache-dir uv==0.11.10
COPY pyproject.toml uv.lock ./
COPY README.md LICENSE ./
COPY LICENSES ./LICENSES
COPY src ./src
RUN uv sync --frozen --no-dev --extra service --no-editable

FROM ${PYTHON_IMAGE} AS controller
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY --from=controller-build /app/.venv /app/.venv
EXPOSE 8000
CMD ["uvicorn", "reliquary_sandbox_service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-proxy-headers"]
