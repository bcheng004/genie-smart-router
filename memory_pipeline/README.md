# Long-term memory Lakeflow pipeline

This is a **Lakeflow Spark Declarative Pipeline**, not a Lakeflow Job or the managed
agent-memory pipeline API. Its graph uses Auto Loader, two streaming tables, expectations,
an append flow, and a Python `foreach_batch_sink`. Model inference and transactional
consolidation execute inside that sink. The long-term store is an external Lakebase table,
not a pipeline-owned Delta table.

The serverless sink uses the pure-Python `pg8000` Postgres driver and authenticated
Databricks SDK calls to Unity Gateway. It deliberately does not load native `libpq`/
`psycopg` or the larger agent SDK dependency stack into the pipeline sandbox. The app
continues to use `psycopg` for its Lakebase sessions.

```text
Application chat settlement
  -> Lakebase chats + memory_outbox (one transaction)
  -> background publisher -> immutable JSON files in a UC Volume
  -> bronze_memory_events -> silver_memory_inputs
  -> consolidate_long_term_memory sink
  -> Lakebase long_term_memories + memory_processed_events (one transaction per event)
  -> actor-scoped recall on the first turn of a new chat
```

## Prerequisites

- A workspace with serverless pipelines and Python foreach-batch sinks available. The
  bundle deliberately uses the `PREVIEW` channel for the sink API.
- An existing Unity Catalog catalog and schema. Do not use a schema that grants broad
  access to other users: source files and both pipeline tables contain private user text.
- The router's existing Lakebase endpoint and database. The database defaults to
  `databricks_postgres` and its Postgres schema defaults to `agent_memory`.
- An explicitly chosen Databricks CLI profile. No profile or workspace is selected by the
  implementation. List available profiles with `databricks auth profiles --skip-validate`.
- A Unity AI Gateway model service that the pipeline identity can query. GPT models use
  the native Responses API; other models use Chat Completions, matching the application.
- The pipeline identity must have a Lakebase Postgres role, endpoint connection access,
  schema usage, and DML access to the memory tables. It must also be able to create the
  schema/tables/index during initial setup, or have an administrator initialize them first.
- The application identity needs `USE CATALOG`, `USE SCHEMA`, and `READ VOLUME` /
  `WRITE VOLUME` on the landing volume. The pipeline identity needs `READ VOLUME` and
  permission to create its streaming tables in the destination schema. Only trusted
  producers may write the volume: actor identity comes from the authenticated app, not
  from the LLM, but someone who can forge source files can forge actor IDs.
- Serverless compute must be able to reach the Lakebase endpoint and Unity Gateway.

## Deploy the pipeline

This is a separate bundle so existing app deployments do not suddenly require new catalog
variables or provision memory resources. Run these commands from `memory_pipeline/`, with
values chosen for your workspace:

```bash
export BUNDLE_VAR_catalog='<existing-catalog>'
export BUNDLE_VAR_schema='<existing-schema>'
export BUNDLE_VAR_lakebase_endpoint='projects/<project>/branches/<branch>/endpoints/<endpoint>'
export BUNDLE_VAR_memory_model='<Unity-Gateway-model-service>'
export BUNDLE_VAR_app_service_principal='<router-app-service-principal-client-id>'

databricks bundle validate --strict -t dev --profile <profile>
databricks bundle deploy -t dev --profile <profile>
databricks pipelines apply-environment <pipeline-id> --profile <profile>
```

Run `apply-environment` after changing dependencies on a warm development pipeline;
otherwise subsequent development updates can continue loading the previously applied
packages despite the new versions appearing in the deployed configuration.

The bundle creates a managed volume named `genie_memory_events` and the `long_term_memory`
pipeline. It does **not** create a job or automatically start inference. Use a separate
destination schema per deployment to avoid sharing private source files across instances.

## Enable the application bridge

The router app includes a **Distill memory** button below its header. It confirms publication,
exports only the selected conversation's eligible user messages for the signed-in actor,
starts the pipeline, and displays progress, failure or completion with a pipeline-details link.
Open a new conversation after completion to use recalled facts. Already exported events are
safe to retry; the button does not delete short-term memory or automatically export other chats.

Configure these environment variables alongside the app's existing `LAKEBASE_ENDPOINT`:

```text
LONG_TERM_MEMORY_VOLUME=/Volumes/<catalog>/<schema>/genie_memory_events
LONG_TERM_MEMORY_PIPELINE_ID=<pipeline-id>
LONG_TERM_MEMORY_AUTO_EXPORT=false
LONG_TERM_MEMORY_SHARED_STORAGE=true
```

The app bundle binds the volume as a `uc_securable` resource and requires the `memory_volume`
(three-part name) and `memory_pipeline_id` variables. It sets automatic export to false;
`shared_memory_storage=true` adds an explicit warning to the confirmation dialog. The pipeline
bundle grants the configured app service principal `CAN_RUN`. The app also needs DML access
to the Lakebase memory tables if a different identity initialized them.

For local development, export the variables in the same shell used to start `app.py` with
your explicitly selected profile. Missing volume/pipeline configuration disables the button.

When the volume is unset, export and recall are disabled. Durable Lakebase sessions must
also be available; in-memory fallback chats are not exported. For trusted private storage,
setting `LONG_TERM_MEMORY_AUTO_EXPORT=true` opts into export on every chat settlement and
publication retries every ten seconds. Manual mode never starts that background publisher.
Both paths create the `events/` directory. Tables initialize lazily; if app and pipeline run under different
identities, initialize with the app identity first and grant the pipeline identity access
to the resulting tables before running inference.

## Backfill existing conversations

From the repository root, export only the explicitly named actor's existing chats:

```bash
.venv/bin/python -m scripts.backfill_memory \
  --profile <profile> \
  --actor '<authenticated-user-email>' \
  --endpoint 'projects/<project>/branches/<branch>/endpoints/<endpoint>' \
  --volume '/Volumes/<catalog>/<schema>/genie_memory_events'
```

The command queues source events and flushes pending exports. It does not invoke a model
or run the pipeline. The invoking identity needs permission to read that actor's stored
chats and write the outbox/volume. Run it only for actors whose histories you are authorized
to process. Legacy transcripts lack message timestamps: backfill uses chat creation time
with a stable per-message ordering offset, not reconstructed real-world timestamps. Legacy
branch copies without source IDs are skipped; backfill the original chat instead.

## Run and smoke-test

From `memory_pipeline/` after deployment and publication:

```bash
databricks bundle run long_term_memory -t dev --profile <profile>
```

This is a triggered pipeline: run another update to process new files. There is no hidden
job schedule. An update uses Auto Loader checkpoints; it does not rescan the Postgres
transcript tables. Source ingestion is capped at ten files per micro-batch, and model calls
are serial and bounded per event. This implementation targets modest chat volume, not
high-throughput multi-tenant ingestion.
The sink collects at most 101 events to the driver and fails before processing if a
micro-batch contains more than 100; it never silently drops excess events. App publication
writes one event per source file, normally keeping a micro-batch within the ten-file limit.

For a live smoke test:

1. In the router app, send: "For TPC-H revenue, always use USD and round to two decimals."
   Wait for the turn to settle, click **Distill memory**, and confirm publication.
2. Follow the button's pipeline-details link. Both streaming tables and the sink
   should succeed. A model/inference/validation failure fails the update rather than
   silently marking the source as processed; already committed events are safe to replay.
3. In the Lakebase SQL editor, inspect the actor's actual memories and processing markers:

   ```sql
   SELECT memory_key, content, evidence, source_id, version
   FROM agent_memory.long_term_memories
   WHERE actor_id = '<authenticated-user-email>' AND NOT deleted;

   SELECT event_id, processed_at
   FROM agent_memory.memory_processed_events
   WHERE actor_id = '<authenticated-user-email>';
   ```

4. Open a **new** conversation and ask which currency you requested for TPC-H revenue.
   Recall is injected as untrusted user-context messages, never as system instructions.
5. Correct the preference in a later message, publish, and run another update. Confirm the
   relevant memory changes and its version increases rather than creating a duplicate.
6. Replay/full-refresh the pipeline: already processed source IDs must not invoke the
   model again or increment memory versions. Also verify another user's new chat does not
   receive the first actor's preferences.

## Guarantees and limits

- Source export includes only user-authored UI messages, not assistant or tool output.
  Messages over 16,000 characters or containing common secret markers are skipped.
  The heuristic is conservative, **not a complete secret/PII detector**.
- The app supplies actor identity. The model may emit only topic-keyed mutations with
  verbatim source evidence. Evidence validation proves a quote exists, not that the model's
  interpretation is semantically correct; evaluate extraction quality with your real model.
- Up to 64 active memories per actor, 1,024 characters per entry, and 16 mutations per
  source event. Invalid output fails processing without advancing its marker.
- A transaction-scoped Postgres advisory lock serializes writers per actor. Memory updates
  and the processed-event marker commit together. A failed transaction can safely retry.
- Idempotency uses actor/event IDs, not Spark batch IDs. External memory is not truncated
  by pipeline full refresh. Model inference itself may repeat if a transaction fails after
  the model responds, but committed memory mutations are not applied twice.
- Newer source timestamps win for the same topic key. Explicit retractions leave
  tombstones so older events cannot resurrect that key. Topic equivalence, deduplication,
  corrections, and deciding which facts deserve memory still depend on the model.
- Newly created branches retain stable message IDs and do not re-export copied messages
  as new observations. The source event stored first remains authoritative.
- "Forget" removes a fact from active recall; it is **not GDPR/retention erasure**. Source
  chats, the outbox, landing files, Delta tables, evidence, and tombstones retain history.
  Define a coordinated retention/erasure policy before production use. Do not independently
  purge landing files if you still need full refresh or historical replay.
- Recall loads at most eight recent active entries at the start of a new chat. It is not
  semantic/vector retrieval, and ongoing chats do not refresh their recalled snapshot.
- This implements extraction, topic-key consolidation and explicit retractions. It is
  not a clone of Databricks' private prompts or autonomous global memory maintenance.

## Local tests

From the repository root, after installing the app's dependencies:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover source filtering, branching, actor isolation, evidence validation, bounded
mutations, corrections, stale replay, tombstones, transactional rollback behavior, export
retry behavior, SQL parameterization, declarative graph registration and application recall.
They use fake repositories and mocked Spark, Postgres and model/network clients: passing
them does not establish live Spark API compatibility, database permissions, model quality
or end-to-end workspace execution. Run the live smoke test after profile/resource selection.

References:
- [Lakeflow Python foreach-batch sinks](https://docs.databricks.com/aws/en/ldp/for-each-batch)
- [Managed agent memory behavior, used as a design reference](https://docs.databricks.com/aws/en/agents/agent-memory/memory-pipelines)
