# Genie Smart Router

Minimal Databricks App (Plotly Dash): an OpenAI Agents SDK agent whose LLM calls go through
Unity AI Gateway
([`databricks_openai.AsyncDatabricksOpenAI`](https://docs.databricks.com/aws/en/agents/custom-agents/author-agent?language=OpenAI)),
talking to two Genie agents as MCP servers, with two routing decisions made per turn:

- **Smart routing (model choice)** — the gateway's task_v2 routing API
  (`/ai-gateway/routing/v1/routes:select`) classifies each prompt and picks the model from the
  workspace's chat models (`agent.select_model`); the app never picks a model itself. The
  decision shows in the chat as a `task_v2` box ahead of the reply.
- **Genie routing (data-source choice)** — a dynamic MCP `tool_filter` on the Genie servers asks
  `system.ai.openjev-qwen35-4b` (Unity AI Gateway System One API) once per run which Genie agent
  — or none — should answer, and exposes only that agent's tools
  (`router.py`; the decision shows as an `openjev` box after the user's message). Falls back to
  all Genie tools when unsure.

- `agent.py` — the agent (model = UC model service, `system.ai.gpt-6-luna` by default, via the
  Responses API; Claude models use Chat Completions plus a small fix flattening reasoning
  blocks), `chat_models()` / `select_model()`, and `MemorySession` — the conversation lives in
  memory per signed-in user (swap in a durable `Session` without touching the routing)
- `genie.py` — the two Genie agents as MCP servers (`/api/2.0/mcp/genie/{space_id}`), authenticated
  **on behalf of the signed-in user** with the token Databricks Apps forwards for the
  `mcp.genie` user API scope — no service-principal fallback
- `router.py` — the openjev Genie router: a System One **`choice` question**
  (`POST /ai-gateway/typesafe/v1/systemone`): an unordered criteria map of the Genie agents plus
  "none", answered with the pick, a confidence and per-option probabilities
- `app.py` — Dash chat page streaming each turn's inner workings: the openjev decision after the
  user's message, every Genie tool call with collapsible args/output, and the `task_v2` model
  routing box ahead of the reply (turns run on their own thread; a 1s poll renders the stream)
- `scripts/create_genie_agents.py` — `uv run create-genie-agents --profile <p>` creates/updates
  the TPC-H and NYC Taxi Genie agents (over `samples.tpch` / `samples.nyctaxi`) and adds them as
  app resources in `databricks.yml`

```bash
databricks bundle deploy -t dev -p <profile>
databricks bundle run genie_smart_router -t dev -p <profile>
```

Locally: `DATABRICKS_CONFIG_PROFILE=<profile> DEV_USER=me@example.com uv run python app.py`.
