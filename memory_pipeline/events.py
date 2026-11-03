"""Stable, user-scoped source events and validated memory mutations."""

import hashlib
import json
import math
import re

MAX_TEXT = 16000
MAX_MEMORIES = 64
KEY_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.:/-]{0,95}\Z")
SECRET_PATTERN = re.compile(
    r"password|passwd|api[ _-]?key|access[ _-]?token|bearer\s+\S+|"
    r"\bsk-[a-zA-Z0-9_-]{12,}|\bAKIA[A-Z0-9]{16}\b|-----BEGIN .*PRIVATE KEY",
    re.IGNORECASE,
)


def contains_secret(text: str) -> bool:
    return bool(SECRET_PATTERN.search(text))


def source_events(actor_id: str, chat: dict) -> list[dict]:
    """Export only user-authored messages, never assistant/tool output or branch copies."""
    result = []
    for position, item in enumerate(chat["events"]):
        if item.get("kind") != "user":
            continue
        text = item.get("text", "")
        if not text.strip() or len(text) > MAX_TEXT or contains_secret(text):
            continue
        if chat.get("branch_of") and not item.get("event_id"):
            continue
        source_id = item.get("event_id") or f"{chat['id']}:{position}"
        event_id = hashlib.sha256(f"{actor_id}\0{source_id}".encode()).hexdigest()
        event = {
            "event_id": event_id,
            "actor_id": actor_id,
            "chat_id": chat["id"],
            "occurred_at": float(item.get("created", chat["created"] + position * 0.000001)),
            "text": text,
        }
        validate_event(event)
        result.append(event)
    return result


def validate_event(event: dict) -> None:
    for field in ("event_id", "actor_id", "chat_id", "text"):
        if not isinstance(event.get(field), str) or not event[field].strip():
            raise ValueError(f"Invalid source field: {field}")
    if not re.fullmatch(r"[a-f0-9]{64}", event["event_id"]):
        raise ValueError("Invalid event identifier")
    timestamp = event.get("occurred_at")
    if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp) or timestamp <= 0):
        raise ValueError("Invalid source timestamp")
    if len(event["text"]) > MAX_TEXT or contains_secret(event["text"]):
        raise ValueError("Source is oversized or contains a suspected secret")


def validate_changes(response: str, event: dict, existing: dict) -> dict:
    """Require bounded mutations, known deletion targets, and verbatim user evidence."""
    changes = json.loads(response)
    if not isinstance(changes, dict) or set(changes) != {"upserts", "deletes"}:
        raise ValueError("Expected upserts and deletes only")
    if not all(isinstance(changes[field], list) for field in changes):
        raise ValueError("Memory mutations must be lists")
    if len(changes["upserts"]) + len(changes["deletes"]) > 16:
        raise ValueError("Too many memory mutations")
    keys = set()
    for operation, mutations in changes.items():
        for mutation in mutations:
            fields = {"key", "content", "evidence"} if operation == "upserts" else {
                "key", "evidence"}
            if not isinstance(mutation, dict) or set(mutation) != fields:
                raise ValueError("Invalid mutation fields")
            key = mutation["key"]
            if not isinstance(key, str) or not KEY_PATTERN.fullmatch(key) or key in keys:
                raise ValueError("Invalid or repeated memory key")
            keys.add(key)
            evidence = mutation["evidence"]
            if (not isinstance(evidence, str) or not evidence.strip()
                    or evidence not in event["text"]):
                raise ValueError("Memory evidence must be a quote from the source user")
            if operation == "deletes":
                if key not in existing:
                    raise ValueError("Cannot delete an unknown memory")
            else:
                content = mutation["content"]
                if (not isinstance(content, str) or not content.strip()
                        or len(content) > 1024 or contains_secret(content)):
                    raise ValueError("Invalid or sensitive memory content")
    active_keys = (set(existing) - {item["key"] for item in changes["deletes"]}) | {
        item["key"] for item in changes["upserts"]}
    if len(active_keys) > MAX_MEMORIES:
        raise ValueError("Actor memory limit exceeded; consolidate existing keys")
    return changes


def process_event(event: dict, repository, distill) -> bool:
    """Atomically consolidate an event once; external replay never uses batch IDs."""
    validate_event(event)
    with repository.transaction(event["actor_id"]):
        if repository.seen(event["actor_id"], event["event_id"]):
            return False
        existing = repository.load(event["actor_id"])
        changes = validate_changes(distill(event["text"], existing), event, existing)
        repository.apply(event, changes)
        repository.mark(event)
    return True
