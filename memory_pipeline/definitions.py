"""Lakeflow graph: incremental events -> validated inputs -> transactional memory sink."""

import os

from pyspark import pipelines as dp
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from memory_pipeline.events import process_event
from memory_pipeline.inference import Distiller

spark = SparkSession.getActiveSession()
SOURCE = spark.conf.get("memory.source_path")
ENDPOINT = spark.conf.get("memory.lakebase_endpoint")
DATABASE = spark.conf.get("memory.database", "databricks_postgres")
SCHEMA = spark.conf.get("memory.schema", "agent_memory")
MODEL = spark.conf.get("memory.model")


@dp.table(name="bronze_memory_events", comment="Immutable user-authored memory source events")
def bronze_memory_events():
    return (spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", "json")
            .option("cloudFiles.maxFilesPerTrigger", "10")
            .schema("event_id STRING, actor_id STRING, chat_id STRING, "
                    "occurred_at DOUBLE, text STRING, _rescued_data STRING")
            .load(SOURCE))


@dp.table(name="silver_memory_inputs", comment="User-scoped memory distillation inputs")
@dp.expect_or_fail("valid_source", """
    event_id RLIKE '^[a-f0-9]{64}$' AND length(trim(actor_id)) > 0
    AND length(trim(chat_id)) > 0 AND length(trim(text)) BETWEEN 1 AND 16000
    AND occurred_at > 0 AND NOT isnan(occurred_at) AND _rescued_data IS NULL
""")
def silver_memory_inputs():
    return spark.readStream.table("bronze_memory_events")


@dp.foreach_batch_sink(name="consolidate_long_term_memory")
def consolidate_long_term_memory(batch, batch_id):
    from databricks.sdk import WorkspaceClient

    from memory_pipeline.connection import connect
    from memory_pipeline.store import PostgresRepository, ensure_tables

    client = WorkspaceClient()
    endpoint = client.postgres.get_endpoint(name=ENDPOINT)
    role = os.environ.get("PGUSER") or os.environ.get("DATABRICKS_CLIENT_ID")
    if not role:
        role = client.current_user.me().user_name
    distill = Distiller(MODEL)
    initialized = False
    rows = batch.orderBy("actor_id", "occurred_at", "event_id").limit(101).collect()
    if len(rows) > 100:
        raise ValueError("Memory micro-batch exceeds 100 events; split the source files and retry")
    for row in rows:
        credential = client.postgres.generate_database_credential(endpoint=ENDPOINT)
        with connect(host=endpoint.status.hosts.host, database=DATABASE, user=role,
                     password=credential.token) as conn:
            if not initialized:
                ensure_tables(conn, SCHEMA)
                initialized = True
            repository = PostgresRepository(conn, SCHEMA)
            event = row.asDict()
            event.pop("_rescued_data", None)
            process_event(event, repository, distill)


@dp.append_flow(target="consolidate_long_term_memory")
def publish_long_term_memories():
    return spark.readStream.table("silver_memory_inputs").select(
        col("event_id"), col("actor_id"), col("chat_id"), col("occurred_at"), col("text"))
