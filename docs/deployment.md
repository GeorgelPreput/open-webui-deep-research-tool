# Production deployment

Deep Research talks to **three independently configured services**:

| Surface | Env prefix | What it does |
|---|---|---|
| Chat LLM provider | `DR_LLM_*` | `/chat/completions`, `/models` |
| Embeddings provider | `DR_EMBEDDINGS_*` | `/embeddings` |
| Open WebUI | `DR_OWUI_*` | retrieval, files, knowledge base, chat persistence |

Each surface has its own base URL and bearer token. Set them independently
so chat and embeddings can hit different providers (often required: chat at
OpenAI / Anthropic-compatible / a local LLM; embeddings at Ollama or a
dedicated embedding service), and so each surface has its own rate-limit
domain and its own blast radius on key compromise.

For the local manual-test stack with everything in one compose project, see
[`deploy/README.md`](../deploy/README.md).

---

## Production configuration patterns

### Pattern A — single provider

OWUI's `/openai` proxy fronts both chat and embeddings, using OWUI's own
admin key for all three surfaces. Smallest config. Trade-off: chat and
embedding traffic share one rate-limit bucket inside OWUI's connection, and
one compromised key has full access.

```bash
DR_OWUI_BASE_URL=http://owui:8080
DR_OWUI_API_KEY=sk-owui-admin

DR_LLM_BASE_URL=http://owui:8080/openai
DR_LLM_API_KEY=sk-owui-admin

DR_EMBEDDINGS_BASE_URL=http://owui:8080/openai
DR_EMBEDDINGS_API_KEY=sk-owui-admin
```

### Pattern B — split provider (recommended)

Chat, embeddings, and OWUI each have their own host **and** their own key.
Three rate-limit domains, three independent secrets, smallest blast radius.

```bash
DR_OWUI_BASE_URL=http://owui:8080
DR_OWUI_API_KEY=sk-owui-admin

DR_LLM_BASE_URL=https://api.openai.com/v1
DR_LLM_API_KEY=sk-openai-...

DR_EMBEDDINGS_BASE_URL=http://ollama:11434
DR_EMBEDDINGS_API_KEY=ollama          # Ollama ignores the value; any non-empty string works
```

### Pattern C — constrained embeddings quota

Use when the embeddings provider has a low TPM/RPM ceiling (free-tier
OpenAI, shared Voyage/Cohere keys, a self-hosted GPU you don't want to
saturate). Pattern B's split, plus throttling on the embeddings client:

```bash
# Same DR_OWUI_*, DR_LLM_*, DR_EMBEDDINGS_BASE_URL/API_KEY as Pattern B, then:

DR_ADVANCED_EMBEDDING_CONCURRENCY=1                 # one in-flight call at a time
DR_EMBEDDINGS_THROTTLE_MAX_REQUESTS_PER_SECOND=2    # token-bucket cap on dispatched calls
DR_EMBEDDINGS_THROTTLE_BATCH_MAX_INPUTS=16          # smaller batches → fewer 413s under TPM
DR_EMBEDDINGS_THROTTLE_MAX_RETRIES=8                # tolerate longer 429 storms
DR_EMBEDDINGS_THROTTLE_MAX_DELAY_SECONDS=120        # backoff ceiling
DR_PERSISTENCE_DISABLE_DURING_DEGRADED=true         # skip KB ingest while throttle is tripped

# Shrink per-cycle embedding workload by narrowing the crawl:
DR_WEB_SEARCH_RESULTS_PER_QUERY=2
DR_WEB_EXTRA_RESULTS_PER_QUERY=1
DR_CYCLES_MAX_CYCLES=8
```

When the embeddings throttle trips into degraded mode the engine keeps
serving the user — it just stops uploading new sources to the OWUI KB until
the bucket recovers, so OWUI's own embedding ingestion doesn't compound the
contention.

---

## Environment variable reference

`Consumer` identifies which component reads the var. `LLM` =
`LLMProviderClient`, `Emb` = `EmbeddingProviderClient`, `OWUI` =
`OWUIClient`, `shim` = the entrypoint shim (OpenAPI server / MCP server /
Function), `Coord` = `Coordinator` startup.

| Env var | Consumer | Typical value | Required | Secret | Failure if wrong |
|---|---|---|---|---|---|
| `DR_LLM_BASE_URL` | LLM | `https://api.openai.com/v1` | yes | no | `ValueError: llm_base_url is required` at startup; `httpx.ConnectError` at first chat call |
| `DR_LLM_API_KEY` | LLM | `sk-...` | yes | **yes** | `ValueError: llm_api_key is required` at startup; 401 on `/chat/completions` |
| `DR_LLM_CHAT_PATH` | LLM | `/chat/completions` (default) | no | no | 404 on every chat call |
| `DR_EMBEDDINGS_BASE_URL` | Emb | `http://ollama:11434` | yes | no | `ValueError: embeddings_base_url is required`; `ConnectError` on first embed |
| `DR_EMBEDDINGS_API_KEY` | Emb | `sk-...` / `ollama` | yes | **yes** | `ValueError: embeddings_api_key is required`; 401 on `/embeddings` |
| `DR_EMBEDDINGS_EMBEDDINGS_PATH` | Emb | `/embeddings` (default) | no | no | 404 on every embed call. The doubled `EMBEDDINGS_` is correct: it's `DR_<group=embeddings>_<field=embeddings_path>` |
| `DR_OWUI_BASE_URL` | OWUI, shim | `http://owui:8080` | yes | no | OWUI calls fail with `ConnectError`; KB / persistence / retrieval all unavailable |
| `DR_OWUI_API_KEY` | shim | `sk-owui-admin` | yes (Docker runtimes) | **yes** | OWUI calls return 401. In the OWUI Function runtime the caller's `Authorization` header takes precedence; the env var is only a fallback |
| `DR_DATA_DIR` | shim | `/data/deep_research` | no (defaults vary) | no | Vocab cache / checkpoints written to default temp path; first run slow, no persistence across container restarts |
| `DR_MODELS_RESEARCH_MODEL` | Coord/LLM | `gemma3:12b` | no | no | `Model not found` on first chat call if the ID is unknown to the provider |
| `DR_MODELS_SYNTHESIS_MODEL` | Coord/LLM | `gemma3:27b` | no | no | Synthesis phase fails with `Model not found` |
| `DR_MODELS_EMBEDDING_MODEL` | Coord/Emb | `nomic-embed-text` | no | no | `Model not found` on first embedding call |
| `DR_ADVANCED_LLM_CONCURRENCY` | LLM | `4` (default) | no | no | Too high → upstream 429s; too low → slow runs |
| `DR_ADVANCED_EMBEDDING_CONCURRENCY` | Emb | `8` (default) | no | no | Too high → embedding 429 storms |
| `DR_EMBEDDINGS_THROTTLE_MAX_REQUESTS_PER_SECOND` | Emb | `0` (off, default) | no | no | None — leave at 0 unless your provider is rate-limited |
| `DR_EMBEDDINGS_THROTTLE_BATCH_MAX_INPUTS` | Emb | `64` (default) | no | no | Provider rejects oversized batches; lower if you see 413/422 on `/embeddings` |
| `DR_EMBEDDINGS_THROTTLE_MAX_RETRIES` | Emb | `5` (default) | no | no | Too low → runs fail under transient 429s |
| `DR_PERSISTENCE_DISABLE_DURING_DEGRADED` | Coord | `false` (default) | no | no | When `true`, KB ingest is skipped while embedding throttle is tripped — recommended for low-TPM keys |
| `DR_WEB_SEARCH_RESULTS_PER_QUERY` | web | `3` (default) | no | no | Higher = more sources but more embedding work per cycle |
| `DR_LOG_LEVEL` / `DR_LOGGING_LEVEL` | log | `INFO` / `DEBUG` | no | no | At `DEBUG` you get per-call HTTP traces with redacted keys |
| `DR_LOG_FORMAT` / `DR_LOGGING_FORMAT` | log | `text` or `json` | no | no | `json` for Loki/ELK/Datadog ingestion |

The OpenAPI Tool and MCP runtimes will refuse to start if any of the four
required URL/key pairs is missing — the `Coordinator` raises immediately
rather than failing mid-run.

---

## Kubernetes manifests

Drop-in for the OpenAPI Tool runtime. The MCP runtime is the same shape with
a different image name and a port change (`9000` instead of `8000`).

### 1. Secrets — one per credential surface

Keep the three keys in three separate `Secret` objects so each can be
rotated independently and so RBAC can restrict per-secret access.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: dr-llm-key            # chat provider
  namespace: deep-research
type: Opaque
stringData:
  api-key: sk-replace-me
---
apiVersion: v1
kind: Secret
metadata:
  name: dr-embeddings-key     # embeddings provider
  namespace: deep-research
type: Opaque
stringData:
  api-key: sk-replace-me
---
apiVersion: v1
kind: Secret
metadata:
  name: dr-owui-key           # Open WebUI admin key
  namespace: deep-research
type: Opaque
stringData:
  api-key: sk-owui-admin
```

Naming convention: `dr-<surface>-key`, all three with the same data key
(`api-key`) so the Deployment env block is uniform.

### 2. ConfigMap — non-secret config

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: dr-config
  namespace: deep-research
data:
  DR_OWUI_BASE_URL:                "http://open-webui.owui.svc.cluster.local:8080"
  DR_LLM_BASE_URL:                 "https://api.openai.com/v1"
  DR_LLM_CHAT_PATH:                "/chat/completions"
  DR_EMBEDDINGS_BASE_URL:          "http://ollama.ml.svc.cluster.local:11434"
  DR_EMBEDDINGS_EMBEDDINGS_PATH:   "/embeddings"
  DR_DATA_DIR:                     "/data/deep_research"
  DR_MODELS_RESEARCH_MODEL:        "gpt-4o-mini"
  DR_MODELS_SYNTHESIS_MODEL:       "gpt-4o"
  DR_MODELS_EMBEDDING_MODEL:       "nomic-embed-text"
  DR_LOG_FORMAT:                   "json"
  DR_LOG_LEVEL:                    "INFO"
```

### 3. PersistentVolumeClaim — vocabulary cache and checkpoints

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: dr-data
  namespace: deep-research
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 5Gi
```

`5Gi` is generous; the vocabulary cache for `nomic-embed-text` is ~150 MB,
checkpoints are a few MB per active conversation.

### 4. Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deep-research-openapi
  namespace: deep-research
spec:
  replicas: 1                       # caches are per-process; >1 replica wastes the warm caches
  selector:
    matchLabels: { app: deep-research-openapi }
  template:
    metadata:
      labels: { app: deep-research-openapi }
    spec:
      containers:
        - name: openapi
          image: deep-research-openapi:latest
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef: { name: dr-config }
          env:
            - name: DR_LLM_API_KEY
              valueFrom: { secretKeyRef: { name: dr-llm-key,        key: api-key } }
            - name: DR_EMBEDDINGS_API_KEY
              valueFrom: { secretKeyRef: { name: dr-embeddings-key, key: api-key } }
            - name: DR_OWUI_API_KEY
              valueFrom: { secretKeyRef: { name: dr-owui-key,       key: api-key } }
          volumeMounts:
            - { name: data, mountPath: /data/deep_research }
          readinessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 5
          resources:
            requests: { cpu: "500m", memory: "1Gi" }
            limits:   { cpu: "2",    memory: "4Gi" }
      volumes:
        - name: data
          persistentVolumeClaim: { claimName: dr-data }
---
apiVersion: v1
kind: Service
metadata:
  name: deep-research-openapi
  namespace: deep-research
spec:
  selector: { app: deep-research-openapi }
  ports:
    - { port: 80, targetPort: 8000 }
```

### Key rotation

Per-surface rotation is a Secret update plus a rollout restart:

```bash
kubectl -n deep-research create secret generic dr-llm-key \
  --from-literal=api-key=sk-new-llm-key \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n deep-research rollout restart deployment/deep-research-openapi
```

The other two surfaces keep serving while the rolling restart picks up the
new value. There is no in-process secret reload; a pod restart is required.

### Helm values

Same shape, condensed:

```yaml
# values.yaml
namespace: deep-research

owui:
  baseUrl: http://open-webui.owui.svc.cluster.local:8080
  existingSecret: dr-owui-key

llm:
  baseUrl: https://api.openai.com/v1
  chatPath: /chat/completions
  existingSecret: dr-llm-key
  models:
    research: gpt-4o-mini
    synthesis: gpt-4o

embeddings:
  baseUrl: http://ollama.ml.svc.cluster.local:11434
  embeddingsPath: /embeddings
  existingSecret: dr-embeddings-key
  model: nomic-embed-text
  throttle:
    maxRPS: 0
    batchMaxInputs: 64

storage:
  dataDir: /data/deep_research
  pvc:
    size: 5Gi

logging:
  level: INFO
  format: json
```

The chart should resolve `existingSecret` to `secretKeyRef` blocks for the
three API keys and project the rest into a `ConfigMap` (or into the
container `env` directly).

---

## Pre-flight validation

Run these from a pod in the same namespace (or from your laptop, with the
right URLs) before the first research call. All three must return `2xx`.

```bash
# 1. Chat provider — list models
curl -fsS "$DR_LLM_BASE_URL/models" \
  -H "Authorization: Bearer $DR_LLM_API_KEY" | jq '.data[].id'

# 2. Embeddings provider — actually embed something
curl -fsS -X POST "$DR_EMBEDDINGS_BASE_URL$DR_EMBEDDINGS_EMBEDDINGS_PATH" \
  -H "Authorization: Bearer $DR_EMBEDDINGS_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$DR_MODELS_EMBEDDING_MODEL\",\"input\":\"hello\"}" \
  | jq '.data[0].embedding | length'

# 3. OWUI knowledge endpoint — auth, base URL, and KB write permission
curl -fsS "$DR_OWUI_BASE_URL/api/v1/knowledge/" \
  -H "Authorization: Bearer $DR_OWUI_API_KEY" | jq 'length'

# 4. OWUI models — confirm research/synthesis/embedding IDs are registered
curl -fsS "$DR_OWUI_BASE_URL/api/models" \
  -H "Authorization: Bearer $DR_OWUI_API_KEY" | jq '.data[].id'
```

Then confirm:

- `DR_MODELS_RESEARCH_MODEL`, `DR_MODELS_SYNTHESIS_MODEL`, and
  `DR_MODELS_EMBEDDING_MODEL` appear verbatim in the `/models` output of
  their respective providers. Keep any `prefix.` segment from the OWUI
  OpenAI-connection `prefix_id`, and any `org/model` slash from
  Ollama/Infinity-style providers.
- The OWUI API key user can see (or create) the chats you want persisted —
  see [Compatibility › chat persistence caveat](./compatibility.md#chat-persistence-caveat).

---

## Security

- **Distinct keys per surface.** Three keys, three rate-limit domains, three
  blast radii. Pattern B / Pattern C above are built around this. Reusing
  one key everywhere also reuses one revocation event.
- **Least privilege per provider.** Where the upstream supports scoped keys
  (OpenAI project-scoped keys, Anthropic workspace keys, OWUI admin vs user
  keys), issue a key scoped to exactly what each client calls:
  - LLM key: `/chat/completions`, `/models`.
  - Embeddings key: `/embeddings`.
  - OWUI key: retrieval, files, knowledge, chats — admin role for KB
    creation, owning the target chats for persistence to land.
- **Key redaction.** Every log line that touches a configured key is
  redacted to first-4 + last-4 (`sk-a…0xyz`); values shorter than 9 chars
  render as `********`. Implemented in
  `deep_research/config/logging.py` (`redact_secret`, `_summarize`) and
  triggered by field names containing `key`, `token`, `secret`, or
  `password`. The OpenAPI server's startup line additionally logs only
  `set` / `unset` for the OWUI key — the value never reaches log archives.
- **Kubernetes secret hygiene.** Always `Secret`, never `ConfigMap`. Limit
  `get`/`list` on Secrets in the `deep-research` namespace via RBAC. Prefer
  an external secret store (External Secrets Operator, sealed-secrets, SOPS,
  Vault Agent Injector) over checking raw `stringData` into git.

---

## Migration from the single-OWUI-route assumption

Earlier deployments routed chat and embeddings through OWUI's `/openai`
proxy with one admin key (`DR_OPENAI_*` style). The package now requires
all three surfaces to be configured independently:

| Old single-route var | Today |
|---|---|
| `DR_OPENAI_BASE_URL` | Set both `DR_LLM_BASE_URL` and `DR_EMBEDDINGS_BASE_URL` — to OWUI's `/openai` to preserve old behaviour, or to separate hosts |
| `DR_OPENAI_API_KEY` | Set both `DR_LLM_API_KEY` and `DR_EMBEDDINGS_API_KEY` — same key to preserve old behaviour, or distinct keys (Pattern B/C) |
| _(implicit OWUI route)_ | `DR_OWUI_BASE_URL` / `DR_OWUI_API_KEY` are now explicit |

The fastest migration is Pattern A (same host and key everywhere); peel
chat and embeddings apart later as load and security posture demand.
