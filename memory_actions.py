"""Nonblocking, actor-scoped manual memory distillation and pipeline progress."""

import logging
import os
import threading
import time

import long_term_memory
import memory

_log = logging.getLogger(__name__)
_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_TERMINAL = {"COMPLETED", "FAILED", "CANCELED"}
_TIMEOUT = 1800


def configured() -> bool:
    return bool(os.environ.get("LONG_TERM_MEMORY_PIPELINE_ID")
                and long_term_memory.source_path() and memory.use_durable())


def status(actor_id: str) -> dict:
    with _lock:
        return dict(_jobs.get(actor_id, {"running": False, "message": "Select a saved chat, "
                                        "then distill its user messages into long-term memory."}))


def _set_status(actor_id: str, **changes) -> None:
    with _lock:
        _jobs.setdefault(actor_id, {}).update(changes)


def _state(update) -> str:
    return getattr(update.state, "value", update.state) or "PENDING"


def _run(actor_id: str, chat_id: str) -> None:
    try:
        client = memory._workspace_client()
        pipeline_id = os.environ["LONG_TERM_MEMORY_PIPELINE_ID"]
        pipeline = client.pipelines.get(pipeline_id)
        if pipeline.latest_updates and _state(pipeline.latest_updates[0]) not in _TERMINAL:
            _set_status(actor_id, running=False, message="The memory pipeline is already busy. "
                        "Wait for it to finish, then try again.")
            return
        count = long_term_memory.queue_chat(actor_id, chat_id)
        _set_status(actor_id, message=f"Publishing {count} eligible user messages…")
        while long_term_memory.publish_pending(actor_id=actor_id, chat_id=chat_id):
            pass
        update_id = client.pipelines.start_update(pipeline_id).update_id
        if not update_id:
            raise RuntimeError("The pipeline did not return an update ID")
        _set_status(actor_id, url=f"{client.config.host.rstrip('/')}/pipelines/"
                    f"{pipeline_id}/updates/{update_id}", message="Pipeline update requested…")
        deadline = time.monotonic() + _TIMEOUT
        wait = threading.Event()
        while time.monotonic() < deadline:
            update = client.pipelines.get_update(pipeline_id, update_id).update
            state = _state(update)
            if state in _TERMINAL:
                message = ("Distillation complete. Open a new chat to recall any extracted facts."
                           if state == "COMPLETED" else
                           f"Distillation {state.lower()}. Open pipeline details, then retry "
                           "after the failure is resolved; exported messages are safe to replay.")
                _set_status(actor_id, running=False, message=message)
                return
            _set_status(actor_id, message=f"Distilling memory · {state.lower().replace('_', ' ')}…")
            wait.wait(5)
        _set_status(actor_id, running=False, message="Stopped monitoring after 30 minutes. "
                    "The update may still be running; check pipeline details before retrying.")
    except ValueError as error:
        _set_status(actor_id, running=False, message=str(error))
    except Exception:
        _log.exception("Manual memory distillation failed")
        _set_status(actor_id, running=False, message="Could not distill memory. Check the app "
                    "logs and pipeline permissions, then retry. No chat history was deleted.")


def start(actor_id: str, chat_id: str) -> None:
    if not actor_id or actor_id == "local-user":
        raise ValueError("Sign in before distilling memory")
    if not chat_id:
        raise ValueError("Select a saved conversation first")
    if not configured():
        raise ValueError("Memory distillation is not configured or Lakebase is unavailable")
    with _lock:
        if _jobs.get(actor_id, {}).get("running"):
            return
        _jobs[actor_id] = {"running": True, "message": "Preparing this conversation…"}
    try:
        threading.Thread(target=_run, args=(actor_id, chat_id), daemon=True,
                         name="memory-distillation").start()
    except Exception:
        _set_status(actor_id, running=False, message="Could not start memory distillation. Retry.")
        raise
