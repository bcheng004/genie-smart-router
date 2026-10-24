# Genie Smart Router

Minimal Databricks App (Plotly Dash): an OpenAI Agents SDK agent whose LLM calls go through
Unity AI Gateway
([`databricks_openai.AsyncDatabricksOpenAI`](https://docs.databricks.com/aws/en/agents/custom-agents/author-agent?language=OpenAI)),
talking to two Genie agents as MCP servers, with two routing decisions made per turn:

- **Smart routing (model choice)** — two routers, picked per user from the header dropdown:
  - **Router: Unity Gateway (task_v2)** (default) — the gateway's task_v2 routing API
    (`/ai-gateway/routing/v1/routes:select`) classifies each prompt and picks the model from the
    workspace's chat models (`agent.select_model`); the app never picks a model itself. The
    decision shows in the chat as a `task_v2` box ahead of the reply.
  - **Router: System One (openjev)** — a System One judge answers a `choice` question
    (`POST /ai-gateway/typesafe/v1/systemone`) over a fixed candidate list — the latest Claude
    Sonnet, Claude Opus, and DeepSeek V4.1 (1017) Flash (`agent.select_model_system_one`); the
    route box names the judge with its confidence. Judges are tried in order:
    `system.ai.openjev-qwen35-4b` (the Genie router's judge) is always provisioned and goes
    first; TypeSafe AI Jev 1.14.0 (`system.ai.typesafeai-jev-1-14-0`, see
    [supported models](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models#typesafe-ai-jev-1140))
    is provider-hosted and only serves once the account enables the externally-hosted TypeSafe AI
    models preview — reorder via `MODEL_ROUTER_SYSTEMONE` to prefer it once enabled. A pick
    routes the turn when it's decisive: confidence ≥ 0.5
    (`MODEL_ROUTER_MIN_CONFIDENCE`) or a ≥ 0.15 probability margin over the runner-up
    (`MODEL_ROUTER_MIN_MARGIN` — openjev's calibration runs low on close-call options);
    otherwise the turn falls back to task_v2.
- **Genie routing (data-source choice)** — a dynamic MCP `tool_filter` on the Genie servers asks
  `system.ai.openjev-qwen35-4b` (Unity AI Gateway System One API) once per run which Genie agent
  — or none — should answer, and exposes only that agent's tools
  (`router.py`; the decision shows as an `openjev` box after the user's message). Falls back to
  all Genie tools when unsure.

- `agent.py` — the agent (model = UC model service, `system.ai.gpt-6-luna` by default, via the
  Responses API; Claude models use Chat Completions plus a small fix flattening reasoning
  blocks), `chat_models()` / `select_model()` / `select_model_system_one()` (the two smart
  routers, dispatched by `route_model`), and `MemorySession` — the in-memory `Session`
  fallback (swap in a durable `Session` without touching the routing)
- `genie.py` — the two Genie agents as MCP servers (`/api/2.0/mcp/genie/{space_id}`), authenticated
  **on behalf of the signed-in user** with the token Databricks Apps forwards for the
  `mcp.genie` user API scope — no service-principal fallback
- `router.py` — the openjev Genie router: a System One **`choice` question**
  (`POST /ai-gateway/typesafe/v1/systemone`): an unordered criteria map of the Genie agents plus
  "none", answered with the pick, a confidence and per-option probabilities
- `app.py` — Dash chat page with a header model-router dropdown (Unity Gateway task_v2 vs the
  System One openjev judge) and a sidebar listing all of the signed-in user's chats (New chat
  button; clicking an entry switches conversations; a reload brings them back), streaming each
  turn's inner workings: the openjev decision after the user's message, every Genie tool call
  with collapsible args/output, and the model routing box (labeled `task_v2` or `systemone · jev`)
  ahead of the reply (turns run on their own thread; a 1s poll renders the stream). Each sidebar
  entry has a branch control (the git icon) that forks the conversation: the new chat starts with
  the full agent memory and transcript, then diverges on its own turns
- `memory.py` — durable agent memory on
  [Lakebase Postgres (Autoscaling)](https://docs.databricks.com/aws/en/oltp/projects): each
  chat's OpenAI input items live in `agent_memory.sessions`, its metadata + rendered transcript
  in `agent_memory.chats` (upserted as turns settle), so conversations — and branches — survive
  app restarts. `LakebaseSession` is an `agent.MemorySession` drop-in (same `Session`
  interface, rows instead of a list). Connections use the platform-injected `PG*`/`LAKEBASE_ENDPOINT`
  env vars and a short-lived OAuth token the app mints itself (`generate_database_credential`),
  cached 30 min of its 1 h validity. When the `postgres` app resource isn't attached, or the
  branch can't be reached at boot, the app falls back to in-memory chats
- `scripts/create_genie_agents.py` — `uv run create-genie-agents --profile <p>` creates/updates
  the TPC-H and NYC Taxi Genie agents (over `samples.tpch` / `samples.nyctaxi`) and adds them as
  app resources in `databricks.yml`

```bash
databricks bundle deploy -t dev -p <profile>
databricks bundle run genie_smart_router -t dev -p <profile>
```

Locally: `DATABRICKS_CONFIG_PROFILE=<profile> DEV_USER=me@example.com uv run python app.py`.

Against Lakebase locally (after deploying the app once, so its service principal created the
`agent_memory` schema):

```bash
export LAKEBASE_ENDPOINT=projects/genie-smart-router-memory/branches/smart-router-memory/endpoints/primary
export LAKEBASE_HOST=ep-bitter-bird-d21lj117.database.us-east-1.cloud.databricks.com
DATABRICKS_CONFIG_PROFILE=<profile> DEV_USER=me@example.com uv run python app.py
```
