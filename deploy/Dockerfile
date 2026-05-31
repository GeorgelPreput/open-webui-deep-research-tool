# Image for BOTH the OpenAPI tool server and the MCP server.
# Same dependencies; the compose file picks the command per service.
FROM python:3.11-slim

WORKDIR /app

# Install the deep_research package with the server + mcp extras. Every runtime
# dependency (numpy, scikit-learn, tiktoken, httpx, fastapi, uvicorn, fastmcp,
# pdfplumber, ...) ships manylinux wheels, so no build toolchain is needed.
COPY pyproject.toml ./
COPY deep_research ./deep_research
RUN pip install --no-cache-dir ".[server,mcp]"

ENV PYTHONUNBUFFERED=1 \
    DR_DATA_DIR=/data/deep_research
RUN mkdir -p /data/deep_research

# OpenAPI tool server -> 8000, MCP server -> 9000. `command:` in docker-compose
# overrides this default (which runs the OpenAPI tool server).
EXPOSE 8000 9000
CMD ["uvicorn", "deep_research.entrypoints.openapi_tool.server:app", \
     "--host", "0.0.0.0", "--port", "8000"]
