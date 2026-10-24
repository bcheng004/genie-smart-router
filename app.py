"""Minimal Dash chat app: the gateway's smart routing picks the model each turn, the openjev
router picks the Genie agent, and the chat streams both decisions plus every Genie tool call
as they happen. Each signed-in user keeps any number of chats — a sidebar lists them all.
Chats are durable when a Lakebase branch is attached (memory.py holds the agent's memory and
the transcript); otherwise they live in process memory (agent.MemorySession) — the routing
never knows the difference.
"""
import asyncio
import logging
import os
import threading
import time
import uuid

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, dcc, html
from flask import request

import agent
import genie
import memory

_log = logging.getLogger(__name__)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True).start()


def _on_loop(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


def _user() -> str:
    """The signed-in user (forwarded by Databricks Apps; DEV_USER locally) — scopes the
    in-memory chats."""
    try:
        return request.headers.get("X-Forwarded-Email") or os.environ.get(
            "DEV_USER", "local-user")
    except RuntimeError:  # no request context (Dash validates the layout at import)
        return "local-user"


def _user_token() -> str | None:
    """The user's forwarded token (Genie on-behalf-of); it rides on the Flask request, which
    the turn thread can't see, so it is captured before the thread starts."""
    try:
        return request.headers.get("X-Forwarded-Access-Token")
    except RuntimeError:  # no request context (unit tests)
        return None


# The signed-in user's chats: {"sessions": [record...], "active": id or None} — one record
# per conversation: {"id", "title", "created", "updated", "session" (the agent's Session),
# "events" (the rendered transcript so far), "branch_of"}. In durable mode (memory.py) each
# record is backed by Lakebase rows and restored from them on process start.
_user_data: dict[str, dict] = {}

# durable mode: which users' persisted chats have been loaded into _user_data this process
_loaded: set[str] = set()
_loaded_lock = threading.Lock()

# Turns run on their own thread (the send callback returns at once); they stream progress —
# the routing decisions, each tool call — into this buffer, which the chat's dcc.Interval
# polls and renders until the turn completes and its events fold into the session record.
# {"sid", "events", "done", "error", "session"}, keyed by user.
_live: dict[str, dict] = {}
_live_lock = threading.Lock()


def _live_event(live: dict, event: dict) -> None:
    """Append one streamed event to the run's buffer."""
    with _live_lock:
        live.setdefault("events", []).append(event)


def _new_chat(text: str, branch_of: str | None = None) -> dict:
    """A fresh conversation record; the message that started it is its title (`branch_of`
    marks a branch — a copy of another chat's memory and transcript). Durable when the
    Lakebase branch is reachable, in-memory otherwise."""
    chat_id = uuid.uuid4().hex[:12]
    session = (memory.new_session(chat_id) if memory.use_durable()
               else agent.MemorySession())
    return {"id": chat_id, "title": text, "created": time.time(),
            "updated": time.time(), "session": session, "events": [],
            "branch_of": branch_of}


def _branch_chat(user: str, src: dict) -> dict:
    """A new conversation that starts as a copy of `src`: the agent keeps the full history,
    the chat shows the same transcript — the two then diverge as each gets its own turns.
    Durable mode copies rows in Lakebase; on a database hiccup it falls back to an
    in-memory session so the branch still works."""
    copy = _new_chat(src["title"] + " (branch)", branch_of=src["id"])
    copy["events"] = list(src["events"])
    if memory.use_durable():
        try:
            memory.copy_sessions(src["id"], copy["id"])
            memory.save_chat(user, copy)
            return copy
        except Exception:  # noqa: BLE001 - a branch should never fail to open
            _log.exception("durable branch copy failed; using an in-memory session")
            copy["session"] = agent.MemorySession()
    items = _on_loop(src["session"].get_items())
    if items:
        _on_loop(copy["session"].add_items(items))
    return copy


def _user_state(user: str) -> dict:
    """The signed-in user's app state: their chats, the active one, and the smart-router
    choice ("task_v2" — the gateway's routing API — or "systemone" — the jev judge)."""
    return _user_data.setdefault(
        user, {"sessions": [], "active": None, "router": "task_v2"})


def _record(user: str, sid: str | None) -> dict | None:
    data = _user_data.get(user)
    return next((r for r in data["sessions"] if r["id"] == sid), None) if data else None


def _settle(user: str) -> None:
    """Fold a finished turn's events into its session transcript — normally the poll does this;
    this catches a finished turn nobody folded yet (e.g. the next message raced the poll).
    Durable mode then upserts the record, so the transcript survives a restart."""
    with _live_lock:
        live = _live.pop(user, None) if (_live.get(user) or {}).get("done") else None
    if live is None:
        return
    rec = _record(user, live["sid"])
    if rec is not None:
        rec["events"].extend(live["events"] + ([] if not live.get("error") else
                                               [{"kind": "reply",
                                                 "text": f"**error:** {live['error']}"}]))
        rec["updated"] = time.time()
        if memory.use_durable():
            try:
                memory.save_chat(user, rec)
            except Exception:  # noqa: BLE001 - transcript persistence is best-effort
                _log.exception("chat transcript save failed")


def _ensure_loaded(user: str) -> None:
    """Durable mode, once per process per user: pull their persisted chats out of Lakebase
    into _user_data (the agent's memory stays in its tables, read through the session)."""
    if not memory.use_durable() or user in _loaded:
        return
    with _loaded_lock:
        if user in _loaded or not memory.use_durable():
            return
        _loaded.add(user)  # even on failure — don't retry against a down DB every request
        try:
            rows = memory.load_chats(user)
        except Exception:  # noqa: BLE001
            _log.exception("chat restore failed; starting with in-memory chats")
            return
        data = _user_state(user)
        if data["sessions"]:  # the user already chatted in this process; keep those
            return
        data["sessions"] = [
            {"id": r["chat_id"], "title": r["title"], "created": r["created"],
             "updated": r["updated"], "branch_of": r["branch_of"], "events": r["events"],
             "session": memory.new_session(r["chat_id"])}
            for r in rows]
        if data["sessions"]:
            data["active"] = data["sessions"][0]["id"]  # pick up the newest chat


def _chat_events(user: str) -> list[dict]:
    """The transcript on screen: the active session's events, plus the in-flight turn's when
    it belongs to this session."""
    rec = _record(user, (_user_data.get(user) or {}).get("active"))
    events = list(rec["events"]) if rec else []
    with _live_lock:
        live = _live.get(user)
    if live and rec and live.get("sid") == rec["id"]:
        events += live["events"]
    return events


def _chat_view(user: str) -> list:
    events = _chat_events(user)
    if not events:
        return [html.Div([html.I(className="bi bi-chat-left-dots me-2"),
                          "Start a chat — ask about TPC-H orders or NYC taxi trips."],
                         className="chat-empty")]
    return _transcript(events)


def _sidebar(user: str) -> list:
    """The chats sidebar: a New chat button, then one entry per conversation, newest first,
    the active one highlighted."""
    data = _user_data.get(user) or {"sessions": [], "active": None}
    items = [html.Button([html.I(className="bi bi-plus-lg me-2"), "New chat"],
                         id="new-chat", className="new-chat-btn", n_clicks=0)]
    for rec in sorted(data["sessions"], key=lambda r: -r["updated"]):
        items.append(html.Div(
            [html.Div(
                [html.Div(rec["title"], className="session-title", title=rec["title"]),
                 html.Div(time.strftime("%b %d, %H:%M", time.localtime(rec["created"])),
                          className="session-time")],
                id={"type": "session-item", "index": rec["id"]}, n_clicks=0,
                className="session-item" + (" active" if rec["id"] == data["active"] else "")),
             html.Button(html.I(className="bi bi-git"), title="Branch this chat",
                         id={"type": "branch-btn", "index": rec["id"]}, n_clicks=0,
                         className="branch-btn")],
            className="session-row"))
    if len(items) == 1:
        items.append(html.Div("No conversations yet", className="sidebar-empty"))
    return items


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
                title="Genie Smart Router")
server = app.server

# probe the Lakebase branch in the background (ready, or fall back to in-memory sessions)
threading.Thread(target=memory.warm, daemon=True, name="memory-warm").start()


def _layout():
    """Per page load: sidebar and active chat render server-side. Chats persist — in
    Lakebase when durable, in process memory otherwise — so a reload brings them back."""
    user = _user()
    data = _user_state(user)
    _ensure_loaded(user)
    with _live_lock:
        live = _live.get(user)
    return html.Div([
        dcc.Interval(id="tick", interval=1000, disabled=not (live and not live.get("done"))),
        html.Div(className="app-shell", children=[
            html.Div(id="sidebar", className="sidebar", children=_sidebar(user)),
            html.Div(className="main", children=[
                html.Div([  # header
                    html.H3("Genie Smart Router", className="app-title"),
                    html.Div([
                        html.Span([html.I(className="bi bi-signpost-split-fill me-1"),
                                   "Smart routing"], className="route-badge",
                                  title="Each turn is routed: a model router picks the chat "
                                        "model, and the openjev router picks the Genie agent"),
                        html.Div(dcc.Dropdown(
                            id="model-router", clearable=False, className="router-select",
                            options=[
                                {"label": "Router: Unity Gateway (task_v2)",
                                 "value": "task_v2"},
                                {"label": "Router: System One (openjev)", "value": "systemone"},
                            ],
                            value=data.get("router") or "task_v2",
                        ), className="router-select-wrap"),
                    ], className="header-right"),
                ], className="app-header"),
                html.Div(id="chat", className="chat", children=_chat_view(user)),
                html.Div([
                    dcc.Input(id="msg", placeholder="Ask about TPC-H orders or NYC taxi trips…",
                              className="msg-input", type="text", debounce=True, n_submit=0),
                    html.Button(html.I(className="bi bi-send-fill"), id="send", n_clicks=0,
                                className="send-btn"),
                ], className="input-bar"),
            ]),
        ]),
    ])


app.layout = _layout


def _pretty(text: str) -> str:
    import json
    try:
        return json.dumps(json.loads(text), indent=2)
    except Exception:  # noqa: BLE001 - model/Genie output need not be JSON
        return text


def _tool_box(call: dict, running: bool = False):
    """One Genie MCP tool call as a box: name, args and output, each collapsible. `running`
    marks a streamed call whose result hasn't arrived yet (spinner)."""
    name = call.get("name") or "tool"
    body = [html.Code(name, className="tool-name")]
    if running:
        body.append(html.I(className="bi bi-arrow-repeat ms-2 tool-spin"))
    if call.get("arguments") and str(call["arguments"]).strip():
        body.append(html.Details([html.Summary("args"), html.Pre(_pretty(str(call["arguments"])))],
                                 className="tool-args"))
    out = call.get("output")
    if out and str(out).strip():
        body.append(html.Details([html.Summary("output"), html.Pre(_pretty(str(out))[:4000])],
                                 className="tool-out"))
    return html.Div(body, className="tool-box")


def _genie_route_box(decision: dict | None) -> html.Div:
    """The openjev router's decision: which Genie agent answers this turn."""
    if decision:
        label = (decision["choice"] or "?").removeprefix("genie_")
        text = f"routed to {label} ({decision['confidence']:.2f})"
    else:
        text = "unsure — model picks the data source"
    return html.Div([html.Code("openjev", className="tool-name"),
                     html.Span([html.I(className="bi bi-database me-1"), text],
                               className="route-text")], className="tool-box route-box")


def _model_route_box(model: str, why: str | None, router: str = "task_v2") -> html.Div:
    """The smart router's decision: which chat model answers this turn, and by which router
    (the gateway's task_v2 API, or a System One judge — the judge and its confidence are in
    `why`)."""
    label = {"task_v2": "task_v2", "systemone": "System One"}.get(router, "model")
    route_text = html.Span(f"routed to {model.removeprefix('system.ai.')}",
                           className="route-text")
    return html.Div([html.Code(label, className="tool-name"), route_text,
                     html.Div(why, className="route-why")] if why else
                    [html.Code(label, className="tool-name"), route_text],
                    className="tool-box route-box")


def _transcript(events: list[dict]) -> list:
    """The conversation as chat elements: routing boxes, tool calls, the replies. A tool
    call without a matching tool_out yet (still running) renders with a spinner."""
    finished = {id(ev) for ev in events if ev["kind"] == "tool_out"}
    out = []
    for i, ev in enumerate(events):
        if ev["kind"] == "user":
            out.append(html.Div(html.Div(ev["text"], className="msg msg-user"),
                                className="msg-row"))
        elif ev["kind"] == "router":
            out.append(_genie_route_box(ev["decision"]))
        elif ev["kind"] == "model":
            out.append(_model_route_box(ev["model"], ev.get("why"), ev.get("router", "task_v2")))
        elif ev["kind"] == "tool":
            done = any(j > i and events[j].get("name") == ev.get("name")
                       and events[j]["kind"] == "tool_out" for j in range(i + 1, len(events)))
            out.append(_tool_box(ev, running=not done))
        elif ev["kind"] == "tool_out":
            out.append(_tool_box(ev))
        elif ev["kind"] == "reply" and ev.get("text", "").strip():
            out.append(html.Div(dcc.Markdown(ev["text"], className="md"),
                                className="msg msg-agent"))
    return out


def _execute_turn(live: dict, text: str, token: str | None):
    """One turn on its own thread: smart-route the model (by the user's chosen router), run
    the agent (the openjev Genie router filters its tools), stream progress into the live
    buffer."""
    try:
        client = genie.user_client(token)
        reply, model_route, _ = _on_loop(agent.run(
            text, session=live["session"], user_client=client,
            model_router=live.get("router") or "task_v2",
            on_event=lambda ev: _live_event(live, ev)))
        _live_event(live, {"kind": "model", "model": model_route[0], "why": model_route[1],
                           "router": model_route[2]})
        _live_event(live, {"kind": "reply", "text": reply})
    except Exception as e:  # noqa: BLE001 - logged, then surfaced in the chat
        _log.exception("turn failed")
        live["error"] = str(e)[:1000]
    finally:
        live["done"] = True


@app.callback(Output("chat", "children"), Output("sidebar", "children"),
              Output("msg", "value"), Output("send", "disabled"), Output("msg", "disabled"),
              Output("tick", "disabled"),
              Input("send", "n_clicks"), Input("msg", "n_submit"), Input("tick", "n_intervals"),
              Input("new-chat", "n_clicks"),
              Input({"type": "session-item", "index": ALL}, "n_clicks"),
              Input({"type": "branch-btn", "index": ALL}, "n_clicks"),
              Input("model-router", "value"),
              State("msg", "value"))
def on_message(_send, _submit, _tick, _new, _items, _branch, router_sel, text):
    """Every chat interaction in one callback — send, poll, New chat, pick a session,
    branch a session, pick the model router — since they all read the same state. `tick`
    renders the in-flight turn each second; when the turn completes, its events fold into
    the session record."""
    user = _user()
    data = _user_state(user)
    _ensure_loaded(user)
    tid = dash.ctx.triggered_id
    with _live_lock:
        live = _live.get(user)
    busy = live is not None and not live.get("done")

    if tid == "model-router":  # smart-router choice; applies from the next turn on
        data["router"] = router_sel or "task_v2"
        return (dash.no_update,) * 6

    if isinstance(tid, dict) and tid.get("type") == "branch-btn":  # fork this conversation
        _settle(user)  # a finished turn the poll hasn't folded yet: fork from a clean state
        src = _record(user, tid["index"])
        if src is None:
            return (dash.no_update,) * 6
        rec = _branch_chat(user, src)
        data["sessions"].append(rec)
        data["active"] = rec["id"]
        return _chat_view(user), _sidebar(user), "", busy, busy, not busy

    if isinstance(tid, dict):  # a sidebar entry: show that conversation
        data["active"] = tid["index"]
        return _chat_view(user), _sidebar(user), "", busy, busy, not busy

    if tid == "new-chat":  # the next message starts a fresh conversation
        _settle(user)
        data["active"] = None
        return _chat_view(user), _sidebar(user), "", busy, busy, not busy

    if tid == "tick":
        if live is None:
            return (dash.no_update,) * 6
        if not live.get("done"):  # the turn is running: render its stream
            return _chat_view(user), dash.no_update, dash.no_update, True, True, False
        _settle(user)  # finished: fold the events in, re-enable the input bar
        return _chat_view(user), _sidebar(user), "", False, False, True

    if not (text or "").strip():
        return (dash.no_update,) * 6
    if busy:  # one turn at a time per user
        return dash.no_update, dash.no_update, dash.no_update, True, True, False
    _settle(user)  # a finished turn the poll hasn't folded yet

    rec = _record(user, data["active"])
    if rec is None:  # no chat on screen (New chat, or the very first message)
        rec = _new_chat(text.strip())
        data["sessions"].append(rec)
        data["active"] = rec["id"]

    token = _user_token()
    new_live = {"sid": rec["id"], "events": [{"kind": "user", "text": text.strip()}],
                "done": False, "error": None, "session": rec["session"],
                "router": data.get("router") or "task_v2"}
    with _live_lock:
        _live[user] = new_live
    threading.Thread(target=_execute_turn, args=(new_live, text.strip(), token),
                     daemon=True).start()
    return _chat_view(user), _sidebar(user), "", True, True, False


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("DATABRICKS_APP_PORT", 8000)))
