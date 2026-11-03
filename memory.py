"""Durable agent memory: chats stored in Lakebase Postgres (Autoscaling).

Two tables in one schema:
- `sessions(chat_id, seq, item)` — the agent's OpenAI input items, one row per item, so the
  agent keeps its history across app restarts (and a branch is a plain row copy).
- `chats(chat_id, ...)` — the chat's metadata + rendered transcript (the sidebar entries and
  event stream), upserted as each turn settles, so the whole UI survives a restart too.

The connection follows the Databricks Apps pattern
(https://docs.databricks.com/aws/en/oltp/projects/authentication): the app config binds
LAKEBASE_ENDPOINT to the `postgres` resource (`value_from` resolves to the endpoint resource
path), and every connection authenticates with a short-lived OAuth token minted through the
app's own identity (`generate_database_credential`), cached for ~30 min of its 1h validity.
The hostname is resolved from the endpoint path (nothing else is auto-injected for
Autoscaling Lakebase) and the Postgres role is the app's service principal
(DATABRICKS_CLIENT_ID, or the signed-in user locally).
Connections are per operation (a turn costs a handful of inserts; one TLS handshake is
noise next to it), which sidesteps stale-pool and token-rotation failure modes.

Locally, set LAKEBASE_ENDPOINT (the endpoint resource path) + LAKEBASE_HOST (its hostname)
to run against a real branch; without them — or when the database can't be reached at boot
— the app falls back to in-memory sessions (`enabled` / `use_durable`).
"""
import asyncio
import functools
import logging
import os
import socket
import subprocess
import threading
import time

import psycopg
from agents.memory import SessionABC
from databricks.sdk import WorkspaceClient
from psycopg.types.json import Json

_log = logging.getLogger(__name__)

SCHEMA = os.environ.get("LAKEBASE_SCHEMA", "agent_memory")
# token validity is 1h; refresh ahead of it (Lakebase guidance: every 30-40 min)
_TOKEN_TTL = 1800.0

_client: WorkspaceClient | None = None
_client_lock = threading.Lock()
_token: dict = {"value": None, "at": 0.0}
_state = "unconfigured"  # unconfigured | ready | unavailable — resolved by warm()
_state_lock = threading.Lock()


def enabled() -> bool:
    """Whether durable memory is configured (platform-injected or explicit local env)."""
    return bool(os.environ.get("LAKEBASE_ENDPOINT"))


def use_durable() -> bool:
    """Whether new chats should be durable: configured, and not known to be down (warm()
    may still be probing — optimistic until proven unavailable)."""
    return enabled() and _state != "unavailable"


def _workspace_client() -> WorkspaceClient:
    global _client
    with _client_lock:
        if _client is None:
            _client = WorkspaceClient()
        return _client


def _credential() -> str:
    """A Lakebase OAuth token (~1h validity), cached and refreshed lazily."""
    if _token["value"] and time.monotonic() - _token["at"] < _TOKEN_TTL:
        return _token["value"]
    endpoint = os.environ.get("LAKEBASE_ENDPOINT")
    if not endpoint:
        raise RuntimeError("LAKEBASE_ENDPOINT is not set; cannot mint a database credential")
    token = _workspace_client().postgres.generate_database_credential(endpoint=endpoint).token
    _token.update(value=token, at=time.monotonic())
    return token


def _refresh_credential() -> None:
    _token.update(value=None, at=0.0)


def _hostaddr(host: str) -> str | None:
    """IP for psycopg's `hostaddr` when resolution fails (macOS getaddrinfo struggles with
    long Lakebase hostnames); `host` still rides along for TLS SNI."""
    try:
        socket.getaddrinfo(host, None)
        return None
    except socket.gaierror:
        try:
            out = subprocess.run(["dig", "+short", host], capture_output=True, text=True,
                                 timeout=10, check=True)
            ips = [l for l in out.stdout.split() if l.strip()]
            return ips[-1] if ips else None
        except Exception:  # noqa: BLE001 - best effort; the connect will error anyway
            return None


@functools.cache
def _endpoint_host() -> str:
    """The endpoint's hostname: PGHOST/LAKEBASE_HOST when set (local runs), else resolved
    from the LAKEBASE_ENDPOINT resource path via the SDK (the deployed-app case — the Apps
    platform injects only the endpoint path, not the host)."""
    if os.environ.get("PGHOST") or os.environ.get("LAKEBASE_HOST"):
        return os.environ.get("PGHOST") or os.environ["LAKEBASE_HOST"]
    endpoint = os.environ.get("LAKEBASE_ENDPOINT")
    if not endpoint:
        raise RuntimeError("no Lakebase host: set PGHOST (or LAKEBASE_HOST), "
                           "or LAKEBASE_ENDPOINT to resolve it from")
    return _workspace_client().postgres.get_endpoint(name=endpoint).status.hosts.host


def _connect():
    kwargs = {"host": _endpoint_host(), "port": int(os.environ.get("PGPORT", "5432")),
              "dbname": os.environ.get("PGDATABASE", "databricks_postgres"),
              "user": (os.environ.get("PGUSER") or os.environ.get("DATABRICKS_CLIENT_ID")
                       or _workspace_client().current_user.me().user_name),
              "password": _credential(), "sslmode": "require"}
    addr = _hostaddr(kwargs["host"])
    if addr:
        kwargs["hostaddr"] = addr
    return psycopg.connect(**kwargs)


def _raw(fn):
    """One DB operation on a bare connection (used before the ready flag is set)."""
    with _connect() as conn:
        return fn(conn)


def _run(fn):
    """One DB operation on a fresh connection; a stale token (auth error) refreshes once
    and retries."""
    _ensure_ready()
    try:
        with _connect() as conn:
            return fn(conn)
    except psycopg.OperationalError:
        _log.warning("lakebase op failed, retrying with a fresh token", exc_info=True)
        _refresh_credential()
        with _connect() as conn:
            return fn(conn)


def _ensure_ready() -> None:
    """Create the schema and tables on first use. The creator owns the schema — for the
    deployed app that's its service principal, which is why the app must deploy before any
    local run against the same branch. A local run finds the tables already there (probe)
    and never needs the DDL — it has DML rights but not ownership."""
    global _state
    if _state == "ready":
        return
    with _state_lock:
        if _state == "ready":
            return

        def probe(conn):
            row = conn.execute("SELECT to_regclass(%s), to_regclass(%s)",
                               (f'"{SCHEMA}".sessions', f'"{SCHEMA}".chats')).fetchone()
            return row[0] is not None and row[1] is not None

        if _raw(probe):
            _state = "ready"
            _log.info("agent memory: lakebase ready (schema %s found)", SCHEMA)
            return

        def create(conn):
            with conn.transaction():
                conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"')
                conn.execute(f'CREATE TABLE IF NOT EXISTS "{SCHEMA}".sessions ('
                             "chat_id text NOT NULL, seq integer NOT NULL, item jsonb NOT NULL, "
                             'PRIMARY KEY (chat_id, seq))')
                conn.execute(f'CREATE TABLE IF NOT EXISTS "{SCHEMA}".chats ('
                             "chat_id text PRIMARY KEY, app_user text NOT NULL, "
                             "title text NOT NULL, created double precision NOT NULL, "
                             "updated double precision NOT NULL, branch_of text, "
                             "events jsonb NOT NULL DEFAULT '[]')")
                conn.execute(f'CREATE INDEX IF NOT EXISTS chats_user_idx '
                             f'ON "{SCHEMA}".chats (app_user, updated)')

        _raw(create)
        _state = "ready"
        _log.info("agent memory: lakebase ready (schema %s created)", SCHEMA)


def warm() -> None:
    """Boot-time probe: resolve ready/unavailable once, so the app knows which memory it's
    on (a failure here falls back to in-memory sessions, never breaks the app)."""
    global _state
    if not enabled():
        return
    try:
        _ensure_ready()
    except Exception:  # noqa: BLE001 - durable memory is best-effort
        _state = "unavailable"
        _log.exception("agent memory: lakebase unavailable, using in-memory sessions")


class LakebaseSession(SessionABC):
    """agent.MemorySession's durable twin: the same Session interface, each OpenAI input
    item a row in the sessions table, keyed by chat."""

    def __init__(self, chat_id: str):
        self._id = chat_id

    async def get_items(self, limit: int | None = None) -> list[dict]:
        return await asyncio.to_thread(
            _run, lambda conn: [
                r[0] for r in conn.execute(
                    f'SELECT item FROM "{SCHEMA}".sessions WHERE chat_id = %s '
                    "ORDER BY seq DESC" + (" LIMIT %s" if limit else ""),
                    (self._id, limit) if limit else (self._id,)).fetchall()[::-1]])

    async def add_items(self, items: list[dict]) -> None:
        if not items:
            return

        def add(conn):
            with conn.transaction():
                seq = conn.execute(
                    f'SELECT COALESCE(MAX(seq), 0) FROM "{SCHEMA}".sessions '
                    "WHERE chat_id = %s", (self._id,)).fetchone()[0]
                for item in items:
                    seq += 1
                    conn.execute(
                        f'INSERT INTO "{SCHEMA}".sessions (chat_id, seq, item) '
                        "VALUES (%s, %s, %s)", (self._id, seq, Json(item)))

        await asyncio.to_thread(_run, add)

    async def pop_item(self) -> dict | None:
        def pop(conn):
            row = conn.execute(
                f'DELETE FROM "{SCHEMA}".sessions WHERE chat_id = %s AND seq = '
                f'(SELECT MAX(seq) FROM "{SCHEMA}".sessions WHERE chat_id = %s) '
                "RETURNING item", (self._id, self._id)).fetchone()
            return row[0] if row else None

        return await asyncio.to_thread(_run, pop)

    async def clear_session(self) -> None:
        await asyncio.to_thread(
            _run, lambda conn: conn.execute(f'DELETE FROM "{SCHEMA}".sessions '
                                            "WHERE chat_id = %s", (self._id,)))


def new_session(chat_id: str) -> LakebaseSession:
    return LakebaseSession(chat_id)


def copy_sessions(src_chat_id: str, dst_chat_id: str) -> None:
    """Branch at the storage layer: copy every item row to the new chat, keeping order."""
    _run(lambda conn: conn.execute(
        f'INSERT INTO "{SCHEMA}".sessions (chat_id, seq, item) '
        f'SELECT %s, seq, item FROM "{SCHEMA}".sessions WHERE chat_id = %s ORDER BY seq',
        (dst_chat_id, src_chat_id)))


def save_chat(user: str, rec: dict) -> None:
    """Upsert a chat's metadata + rendered transcript (the agent's items live in sessions)."""
    from long_term_memory import auto_export_enabled
    from memory_pipeline.events import source_events
    from memory_pipeline.store import enqueue, ensure_tables

    export = auto_export_enabled()

    def save(conn):
        with conn.transaction():
            conn.execute(
                f'INSERT INTO "{SCHEMA}".chats (chat_id, app_user, title, created, updated, '
                "branch_of, events) VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (chat_id) DO UPDATE SET title = EXCLUDED.title, "
                "updated = EXCLUDED.updated, branch_of = EXCLUDED.branch_of, "
                "events = EXCLUDED.events",
                (rec["id"], user, rec["title"], rec["created"], rec["updated"],
                 rec.get("branch_of"), Json(rec["events"])))
            if export:
                ensure_tables(conn, SCHEMA)
                enqueue(conn, SCHEMA, source_events(user, rec))

    _run(save)


def load_chats(user: str) -> list[dict]:
    """The user's persisted chats, newest first (jsonb columns arrive already parsed)."""
    def load(conn):
        rows = conn.execute(
            f'SELECT chat_id, title, created, updated, branch_of, events '
            f'FROM "{SCHEMA}".chats WHERE app_user = %s ORDER BY updated DESC',
            (user,)).fetchall()
        return [{"chat_id": r[0], "title": r[1], "created": r[2], "updated": r[3],
                 "branch_of": r[4], "events": r[5]} for r in rows]

    return _run(load)
