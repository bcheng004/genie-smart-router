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
- `scripts/create_genie_agents.py` creates/updates two Genie agents and configures their app
  resources, environment bindings and routing mode. `--dataset samples` retains the original
  TPC-H / NYC Taxi setup; `--dataset asana --catalog <catalog> --schema <schema>` uses the
  synthetic Asana revenue and marketing tables. Every example SQL and snippet is validated
  before creating or updating agents. Existing agents are matched by title on reruns.

## Long-Term Memory Panel

The right-hand **Long-term memory** panel lists the signed-in user's most recent 50 active
saved facts, with their keys, versions and source timestamps. Refresh manually or finish a
distillation to reload it. Facts are displayed as plain text, with actor-scoped queries and
secret filtering; other users' facts and deleted memories are never included. On smaller
screens the panel moves below the chat. Storage failures show a retry message instead of
being mistaken for an empty memory store.

## Synthetic Asana Demo

This is **fictional Asana-style SaaS data**, not Asana financial results, actual customers,
real prices or forecasts. The approved deployment uses `workshop_guy_catalog.asana_demo`
with the `mlflow-workshop` profile. The catalog's existing access policy applies; no real
customer information should be loaded into this demo schema.

- `revenue`: 120,000 account-month snapshots for 10,000 fictional accounts, November 2025
  through October 2026. Includes invented plans, seats, regions, acquisition campaigns,
  subscription revenue, opening/closing MRR, ARR, expansion and churn.
- `marketing`: 109,500 campaign-day records for 300 campaigns over the same period. Includes
  spend, impressions, clicks, leads, qualified leads, opportunities, acquired accounts and
  attributed new MRR. A simulated Paid Social lead-quality decline increases spending and
  acquisition costs in July–August 2026, shifting some signups to September–October.
- MRR/ARR and seats are snapshots: aggregate across accounts within a month, not across
  months. Synthetic subscription revenue equals closing MRR without proration and is not
  GAAP recognized revenue. Marketing conversions and attributed MRR reconcile to revenue's
  acquisition cohorts. Never join these fact tables without first aggregating to the same
  campaign/month grain. The MRR-to-spend ratio is not realized ROAS or lifetime-value ROI.

Generate using an isolated Python 3.12 environment and Databricks Connect serverless:

```bash
uv run --no-project --python 3.12 \
  --with 'databricks-connect>=17.3,<17.4' --with faker \
  --with 'pandas>=2.2,<3' --with 'pyarrow>=15,<20' \
  python scripts/generate_asana_data.py \
  --profile mlflow-workshop --catalog workshop_guy_catalog --schema asana_demo

uv run create-genie-agents --profile mlflow-workshop \
  --dataset asana --catalog workshop_guy_catalog --schema asana_demo \
  --warehouse-id 9bbc027402111984
```

The generator creates the schema if missing and refuses to replace existing tables unless
`--overwrite` is explicitly supplied. IDs, names and distributions are seeded for reproducibility.
Generation libraries are local tooling, not app runtime dependencies. `--dry-run` prints Genie
payloads without writes; `--validate` checks only SQL. Switching datasets removes only the
other mode's app bindings, preserving Lakebase and memory-pipeline resources; existing Genie
spaces and conversation history are not deleted. Start a new chat after switching datasets.

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

## Revenue custom-claim view

Keep the existing catalog and base table, and expose an admin-only dynamic view:

```sql
CREATE VIEW workshop_guy_catalog.asana_demo.revenue_by_claim AS
SELECT * FROM workshop_guy_catalog.asana_demo.revenue
WHERE current_oauth_custom_identity_claim() = 'admin';
```

`custom_claim=admin` returns revenue rows; `user`, missing (`NULL`), empty and unknown claims
return no rows. The claim belongs to the caller's OAuth token, not a column in the data.
The setup command previews SQL by default. Add `--apply` to create the view; it refuses to
replace an existing object. It does not change base tables or grants.

```bash
.venv/bin/python -m scripts.setup_revenue_claims --profile mlflow-workshop \
  --warehouse-id 9bbc027402111984 --catalog workshop_guy_catalog --schema asana_demo --apply
.venv/bin/python -m scripts.setup_revenue_claims --profile mlflow-workshop \
  --warehouse-id 9bbc027402111984 --catalog workshop_guy_catalog --schema asana_demo --marketing --apply
```

For live admin/user/absent/unknown-claim verification, supply an explicitly chosen service
principal's OAuth client ID and secret through `REVENUE_CLAIMS_CLIENT_ID` and
`REVENUE_CLAIMS_CLIENT_SECRET`, then run the same command with `--verify-claims` **instead of**
`--apply`. The principal needs warehouse `CAN_USE`, catalog/schema usage and `SELECT` on
the view. Credentials and issued tokens are never printed by the verifier. This operator-only
utility intentionally mints test roles; it is not an application endpoint or a user role picker.
Add `--marketing` during verification to also check that `marketing_by_claim` keeps marketing
rows visible but masks `attributed_new_mrr_usd` for every non-admin claim.

The Asana app enables `GENIE_CUSTOM_CLAIMS=true`. Only the verified signed-in identity
`bo.cheng@databricks.com` receives the `admin` claim; every other identity receives `user`.
The app checks the forwarded email against the user token's current-user identity before
minting an OAuth token with its auto-injected `DATABRICKS_CLIENT_ID` and
`DATABRICKS_CLIENT_SECRET`. Missing identities, credentials or token errors fail closed;
there is no unclaimed-token fallback and no browser-controlled role selector.

In this mode the tools use the documented Genie Conversation API rather than relying on
undocumented claim propagation through MCP. Each tool invocation starts a new conversation,
and the app never accepts an externally supplied Genie conversation ID. Model/tool routing
remains enabled. Marketing uses `marketing_by_claim`, which masks revenue-derived new MRR
for non-admins. The revenue tool additionally denies non-admin calls without querying Genie.

To create the two separate claims-aware Genie spaces and update the bundle bindings, provide
the operator verification credentials above and run:

```bash
.venv/bin/python -m scripts.create_genie_agents --profile mlflow-workshop \
  --dataset asana --catalog workshop_guy_catalog --schema asana_demo \
  --warehouse-id 9bbc027402111984 --claim-views
```

This validates examples with an admin-claim token. It does not modify the original raw-table
Genie spaces. Omitting `--claim-views` returns to the original on-behalf-of mode and spaces.
For local claim-aware app development, set the explicit profile, `DEV_USER` to that profile's
real signed-in email, `GENIE_DATASET=asana`, `GENIE_CUSTOM_CLAIMS=true`, both claims-space IDs,
and the operator verification credentials. A fake `DEV_USER` cannot impersonate the admin.

Chats, tool outputs and long-term memories use a new actor namespace including the email,
policy version and server-assigned role. Old memories and chats remain stored unchanged but
are not automatically loaded into claims-mode sessions. Role changes cannot replay another
role's history. Start a fresh chat after deploying this mode.

**Boundary:** the view protects queries against the view, not queries against `revenue`.
The shared catalog currently grants `account users` broad inherited privileges, including to
the app principal. Those grants remain unchanged, so this is not a restriction on all revenue
access. Do not advertise it as such. A principal with base-table privileges can still query
those tables outside the configured views. The underlying marketing table is unchanged too.
The linked Databricks guide recommends ABAC for Genie authoring because an author without
a claim sees no rows in a dynamic view. This setup follows the requested view approach;
author and validate these spaces through the API with an admin-claim token.
See [OAuth custom identity claims](https://docs.databricks.com/aws/en/dev-tools/auth/oauth-custom-identity-claims).

## Long-term memory pipeline

`memory_pipeline/` adds a separate **Lakeflow Spark Declarative Pipeline** that learns
durable facts from user messages and consolidates them into actor-scoped Lakebase memories.
The **Distill memory** button below the app header publishes only the signed-in user's
selected conversation after confirmation, triggers the pipeline, and reports progress/errors.
Automatic export is disabled in the app bundle; new conversations recall extracted facts.
Set bundle variables `memory_volume` (three-part UC volume name) and `memory_pipeline_id`
before deploying. Existing app names can be preserved with the `app_name` variable.
See [pipeline setup, deployment, limits, and live smoke tests](memory_pipeline/README.md).
Run local tests with `.venv/bin/python -m unittest discover -s tests -v`.
