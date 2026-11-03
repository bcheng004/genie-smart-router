"""Opt-in, durable Lakebase -> Volume bridge and actor-scoped memory recall."""

import io
import json
import logging
import os
import threading

from databricks.sdk.errors import AlreadyExists, ResourceAlreadyExists
from psycopg import sql

from memory_pipeline.events import contains_secret, source_events
from memory_pipeline.store import enqueue, ensure_tables

_log = logging.getLogger(__name__)
_publisher_lock = threading.Lock()
_started = False


def source_path() -> str:
    volume = os.environ.get("LONG_TERM_MEMORY_VOLUME", "").rstrip("/")
    if not volume:
        return ""
    parts = volume.split("/")
    if (len(parts) != 5 or parts[:2] != ["", "Volumes"] or not all(parts[2:])
            or any(part in {".", ".."} for part in parts[2:])):
        raise ValueError("LONG_TERM_MEMORY_VOLUME must be /Volumes/<catalog>/<schema>/<volume>")
    return volume + "/events"


def auto_export_enabled() -> bool:
    return bool(source_path()) and os.environ.get(
        "LONG_TERM_MEMORY_AUTO_EXPORT", "true").lower() == "true"


def warm() -> None:
    import memory

    if not source_path() or not memory.use_durable():
        return
    try:
        memory._run(lambda conn: ensure_tables(conn, memory.SCHEMA))
    except Exception:
        _log.exception("Long-term memory table initialization failed")


def queue_chat(actor_id: str, chat_id: str) -> int:
    import memory

    if not actor_id or not chat_id:
        raise ValueError("Choose one of your saved conversations first")
    if not source_path() or not memory.use_durable():
        raise ValueError("Long-term memory requires a configured volume and durable chats")

    def queue(conn):
        with conn.transaction():
            row = conn.execute(sql.SQL("""
                SELECT created, branch_of, events FROM {} WHERE chat_id = %s AND app_user = %s
            """).format(sql.Identifier(memory.SCHEMA, "chats")), (chat_id, actor_id)).fetchone()
            if row is None:
                raise ValueError("Conversation not found for the signed-in user")
            events = source_events(actor_id, {"id": chat_id, "created": row[0],
                                             "branch_of": row[1], "events": row[2]})
            if not events:
                raise ValueError("This conversation has no eligible user messages to distill")
            ensure_tables(conn, memory.SCHEMA)
            enqueue(conn, memory.SCHEMA, events)
            return len(events)

    return memory._run(queue)


def publish_pending(limit: int = 100, actor_id: str | None = None,
                    chat_id: str | None = None) -> int:
    import memory

    path = source_path()
    if not path or not memory.use_durable():
        return 0
    if chat_id and not actor_id:
        raise ValueError("Publishing a conversation requires its authenticated actor")
    client = memory._workspace_client()
    client.files.create_directory(path)

    def publish(conn):
        ensure_tables(conn, memory.SCHEMA)
        with conn.transaction():
            filters = sql.SQL("")
            params = []
            if actor_id:
                filters += sql.SQL(" AND actor_id = %s")
                params.append(actor_id)
            if chat_id:
                filters += sql.SQL(" AND payload ->> 'chat_id' = %s")
                params.append(chat_id)
            rows = conn.execute(sql.SQL("""
                SELECT event_id, payload FROM {} WHERE NOT published {}
                ORDER BY event_id LIMIT %s FOR UPDATE SKIP LOCKED
            """).format(sql.Identifier(memory.SCHEMA, "memory_outbox"), filters),
                                (*params, limit)).fetchall()
            for event_id, payload in rows:
                content = json.dumps(payload, allow_nan=False).encode() + b"\n"
                try:
                    client.files.upload(f"{path}/{event_id}.json", io.BytesIO(content),
                                        overwrite=False)
                except (AlreadyExists, ResourceAlreadyExists):
                    pass
                conn.execute(sql.SQL("UPDATE {} SET published = true WHERE event_id = %s")
                             .format(sql.Identifier(memory.SCHEMA, "memory_outbox")), (event_id,))
            return len(rows)

    return memory._run(publish)


def start_publisher() -> None:
    global _started
    if not auto_export_enabled():
        return
    with _publisher_lock:
        if _started:
            return
        _started = True

    def run():
        wait = threading.Event()
        while True:
            try:
                publish_pending()
            except Exception:
                _log.exception("long-term memory publication failed; outbox retained for retry")
            wait.wait(10)

    threading.Thread(target=run, daemon=True, name="memory-publisher").start()


def list_memories(actor_id: str, limit: int = 50) -> list[dict]:
    import memory

    if not actor_id:
        raise ValueError("Listing memories requires an authenticated actor")
    if not source_path() or not memory.use_durable():
        return []
    limit = min(max(limit, 1), 100)

    def load(conn):
        return conn.execute(sql.SQL("""
            SELECT memory_key, content, version, source_at FROM {}
            WHERE actor_id = %s AND NOT deleted
            ORDER BY source_at DESC, memory_key LIMIT %s
        """).format(sql.Identifier(memory.SCHEMA, "long_term_memories")),
                            (actor_id, limit)).fetchall()

    return [{"key": key, "content": content, "version": version, "source_at": source_at}
            for key, content, version, source_at in memory._run(load)
            if not contains_secret(content)]


def recall(actor_id: str, limit: int = 8) -> list[dict]:
    import memory

    if not source_path() or not memory.use_durable():
        return []
    if not actor_id:
        raise ValueError("Memory recall requires an authenticated actor")
    limit = min(max(limit, 1), 16)
    try:
        def load(conn):
            return conn.execute(sql.SQL("""
                SELECT memory_key, content FROM {} WHERE actor_id = %s AND NOT deleted
                ORDER BY source_at DESC, memory_key LIMIT %s
            """).format(sql.Identifier(memory.SCHEMA, "long_term_memories")),
                                (actor_id, limit)).fetchall()

        return [{"key": key, "content": content} for key, content in memory._run(load)
                if not contains_secret(content)]
    except Exception:
        _log.exception("long-term memory recall unavailable")
        return []
