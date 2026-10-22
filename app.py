"""Minimal Dash chat app: the gateway's smart routing picks the model each turn, the openjev
router picks the Genie agent, and the chat streams both decisions plus every Genie tool call
as they happen. Conversation history lives in memory per signed-in user (agent.MemorySession).
"""
import asyncio
import logging
import os
import threading

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
from flask import request

import agent
import genie

_log = logging.getLogger(__name__)

_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True).start()


def _on_loop(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


def _user() -> str:
    """The signed-in user (forwarded by Databricks Apps; DEV_USER locally) — scopes the
    in-memory conversation."""
    return request.headers.get("X-Forwarded-Email") or os.environ.get("DEV_USER", "local-user")


def _user_token() -> str | None:
    """The user's forwarded token (Genie on-behalf-of); it rides on the Flask request, which
    the turn thread can't see, so it is captured before the thread starts."""
    try:
        return request.headers.get("X-Forwarded-Access-Token")
    except RuntimeError:  # no request context (unit tests)
        return None


# one in-memory conversation per signed-in user, for the life of the process
_sessions: dict[str, agent.MemorySession] = {}

# Turns run on their own thread (the send callback returns at once); they stream progress —
# the routing decisions, each tool call — into this buffer, which the chat's dcc.Interval
# polls and renders until the turn completes. Keyed by user.
_live: dict[str, dict] = {}
_live_lock = threading.Lock()


def _live_event(live: dict, event: dict) -> None:
    """Append one streamed event to the run's buffer."""
    with _live_lock:
        live.setdefault("events", []).append(event)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
                title="Genie Smart Router")
server = app.server

app.layout = html.Div([
    dcc.Interval(id="tick", interval=1000, disabled=True),
    html.Div([  # header
        html.H3("Genie Smart Router", className="app-title"),
        html.Span([html.I(className="bi bi-signpost-split-fill me-1"), "Smart routing"],
                  className="route-badge",
                  title="Always on: the gateway's task_v2 router picks the model for each "
                        "prompt, and the openjev router picks the Genie agent"),
    ], className="app-header"),
    html.Div(id="chat", className="chat"),
    html.Div([
        dcc.Input(id="msg", placeholder="Ask about TPC-H orders or NYC taxi trips…",
                  className="msg-input", type="text", debounce=True, n_submit=0),
        html.Button(html.I(className="bi bi-send-fill"), id="send", n_clicks=0,
                    className="send-btn"),
    ], className="input-bar"),
])


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [] if isinstance(p, dict))


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


def _model_route_box(model: str, why: str | None) -> html.Div:
    """The smart router's decision: which chat model answers this turn."""
    return html.Div([html.Code("task_v2", className="tool-name"),
                     html.Span(f"routed to {model.removeprefix('system.ai.')}",
                               className="route-text"),
                     html.Div(why, className="route-why")] if why else
                    [html.Code("task_v2", className="tool-name"),
                     html.Span(f"routed to {model.removeprefix('system.ai.')}",
                               className="route-text")], className="tool-box route-box")


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
            out.append(_model_route_box(ev["model"], ev.get("why")))
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
    """One turn on its own thread: smart-route the model, run the agent (the openjev Genie
    router filters its tools), stream progress into the live buffer."""
    try:
        client = genie.user_client(token)
        reply, model_route, _ = _on_loop(agent.run(
            text, session=live["session"], user_client=client,
            on_event=lambda ev: _live_event(live, ev)))
        _live_event(live, {"kind": "model", "model": model_route[0], "why": model_route[1]})
        _live_event(live, {"kind": "reply", "text": reply})
    except Exception as e:  # noqa: BLE001 - logged, then surfaced in the chat
        _log.exception("turn failed")
        live["error"] = str(e)[:1000]
    finally:
        live["done"] = True


@app.callback(Output("chat", "children"), Output("msg", "value"), Output("send", "disabled"),
              Output("msg", "disabled"), Output("tick", "disabled"),
              Input("send", "n_clicks"), Input("msg", "n_submit"), Input("tick", "n_intervals"),
              State("msg", "value"))
def on_message(_send, _submit, _tick, text):
    user = _user()
    if dash.ctx.triggered_id == "tick":  # a turn is running (or just finished): poll it
        with _live_lock:
            live = _live.get(user)
        if live is None or not live.get("done"):
            return (dash.no_update,) * 5
        with _live_lock:
            _live.pop(user, None)
        events = live["events"] + ([] if not live.get("error") else
                                   [{"kind": "reply", "text": f"**error:** {live['error']}"}])
        return _transcript(events), "", False, False, True
    if not (text or "").strip():
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    with _live_lock:  # one turn at a time per user
        live = _live.get(user)
    if live is not None and not live.get("done"):
        return dash.no_update, dash.no_update, True, True, False
    token = _user_token()
    live = {"events": [{"kind": "user", "text": text.strip()}], "done": False, "error": None,
            "session": _sessions.setdefault(user, agent.MemorySession())}
    with _live_lock:
        _live[user] = live
    threading.Thread(target=_execute_turn, args=(live, text.strip(), token),
                     daemon=True).start()
    return _transcript(live["events"]), "", True, True, False


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("DATABRICKS_APP_PORT", 8000)))
