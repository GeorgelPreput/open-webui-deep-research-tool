FROM python:3.11-slim AS build

WORKDIR /src
COPY pyproject.toml README.md ./
COPY deep_research deep_research
# --target installs the project + extras to a relocatable prefix that the
# runtime stage drops straight into site-packages, so no editable install
# and no compiler toolchain ends up in the final image.
RUN pip install --no-cache-dir --target=/install ".[server]"


FROM python:3.11-slim

RUN groupadd --system --gid 10001 appgroup \
 && useradd  --system --uid 10001 --gid appgroup --home-dir /app appuser

WORKDIR /app
COPY --from=build /install /usr/local/lib/python3.11/site-packages/
COPY --from=build /install/bin/ /usr/local/bin/
RUN chown -R appuser:appgroup /app

USER appuser
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "deep_research.entrypoints.openapi_tool.server:app", "--host", "0.0.0.0", "--port", "8000"]
