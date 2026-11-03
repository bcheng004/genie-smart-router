"""Lakebase transactions, provenance, replay markers, and export outbox."""

from contextlib import contextmanager
import json


def identifier(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name:
        raise ValueError("Invalid SQL identifier")
    return '"' + name.replace('"', '""') + '"'


def table(schema: str, name: str) -> str:
    return f"{identifier(schema)}.{identifier(name)}"


def ensure_tables(conn, schema: str) -> None:
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {identifier(schema)}")
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table(schema, "memory_outbox")} (
            event_id text PRIMARY KEY, actor_id text NOT NULL,
            payload jsonb NOT NULL, published boolean NOT NULL DEFAULT false
        )
    """)
    index = conn.execute("SELECT to_regclass(%s)",
                         (table(schema, "memory_outbox_pending_idx"),)).fetchone()[0]
    if index is None:
        conn.execute("CREATE INDEX IF NOT EXISTS memory_outbox_pending_idx "
                     f"ON {table(schema, 'memory_outbox')} (event_id) WHERE NOT published")
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table(schema, "long_term_memories")} (
            actor_id text NOT NULL, memory_key text NOT NULL, content text NOT NULL,
            evidence text NOT NULL, source_id text NOT NULL, source_at double precision NOT NULL,
            deleted boolean NOT NULL DEFAULT false, version integer NOT NULL DEFAULT 1,
            PRIMARY KEY (actor_id, memory_key)
        )
    """)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table(schema, "memory_processed_events")} (
            actor_id text NOT NULL, event_id text NOT NULL,
            processed_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (actor_id, event_id)
        )
    """)


def enqueue(conn, schema: str, events: list[dict]) -> None:
    for event in events:
        conn.execute(f"""
            INSERT INTO {table(schema, "memory_outbox")} (event_id, actor_id, payload)
            VALUES (%s, %s, %s::jsonb)
            ON CONFLICT (event_id) DO NOTHING
        """, (event["event_id"], event["actor_id"], json.dumps(event, allow_nan=False)))


class PostgresRepository:
    def __init__(self, conn, schema: str):
        self.conn = conn
        self.schema = schema

    @contextmanager
    def transaction(self, actor_id: str):
        with self.conn.transaction():
            self.conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                              (f"agent-memory:{self.schema}:{actor_id}",))
            yield

    def seen(self, actor_id: str, event_id: str) -> bool:
        return self.conn.execute(f"""
            SELECT 1 FROM {table(self.schema, "memory_processed_events")}
            WHERE actor_id = %s AND event_id = %s
        """,
                                 (actor_id, event_id)).fetchone() is not None

    def load(self, actor_id: str) -> dict:
        rows = self.conn.execute(f"""
            SELECT memory_key, content FROM {table(self.schema, "long_term_memories")}
            WHERE actor_id = %s AND NOT deleted
            ORDER BY memory_key
        """, (actor_id,)).fetchall()
        return dict(rows)

    def apply(self, event: dict, changes: dict) -> None:
        for operation, mutations in changes.items():
            for mutation in mutations:
                self.conn.execute(f"""
                    INSERT INTO {table(self.schema, "long_term_memories")} AS memories
                        (actor_id, memory_key, content, evidence, source_id, source_at, deleted)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (actor_id, memory_key) DO UPDATE SET
                        content = EXCLUDED.content, evidence = EXCLUDED.evidence,
                        source_id = EXCLUDED.source_id, source_at = EXCLUDED.source_at,
                        deleted = EXCLUDED.deleted, version = memories.version + 1
                    WHERE (EXCLUDED.source_at, EXCLUDED.source_id)
                        > (memories.source_at, memories.source_id)
                """, (
                    event["actor_id"], mutation["key"], mutation.get("content", ""),
                    mutation["evidence"], event["event_id"], event["occurred_at"],
                    operation == "deletes",
                ))

    def mark(self, event: dict) -> None:
        self.conn.execute(f"""
            INSERT INTO {table(self.schema, "memory_processed_events")} (actor_id, event_id)
            VALUES (%s, %s)
        """,
                          (event["actor_id"], event["event_id"]))
