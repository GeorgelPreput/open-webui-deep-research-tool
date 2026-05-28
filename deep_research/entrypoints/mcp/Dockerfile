FROM python:3.11-slim AS build

WORKDIR /src
COPY pyproject.toml README.md ./
COPY deep_research deep_research
RUN pip install --no-cache-dir --target=/install ".[mcp]"


FROM python:3.11-slim

RUN groupadd --system --gid 10001 appgroup \
 && useradd  --system --uid 10001 --gid appgroup --home-dir /app appuser

WORKDIR /app
COPY --from=build /install /usr/local/lib/python3.11/site-packages/
COPY --from=build /install/bin/ /usr/local/bin/
RUN chown -R appuser:appgroup /app

USER appuser
ENV PYTHONUNBUFFERED=1
EXPOSE 9000

CMD ["python", "-m", "deep_research.entrypoints.mcp.server"]
