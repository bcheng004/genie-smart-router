"""Genie routing as part of the OpenAI Agents SDK run.

The agent always carries both Genie MCP servers; `genie_tool_filter` is their dynamic MCP
`tool_filter`, so the SDK asks it which Genie tools to expose whenever it lists tools for the
agent. The first time in a run it asks a small System One model —
`POST <host>/ai-gateway/typesafe/v1/systemone` with `system.ai.openjev-qwen35-4b`
(https://docs.databricks.com/aws/en/machine-learning/model-serving/query-systemone-models),
through the agent's own `databricks_openai.AsyncDatabricksOpenAI` client — a `choice` question:
which Genie agent (or none) should answer the latest message, given the recent conversation.
Only that server's tools are then visible to the model; the decision lives on the run context.

The call runs with the app's identity (it only picks a tool and reads no data; Genie itself still
runs on behalf of the user). If the router is unavailable or not confident enough, every Genie
tool stays visible and the main model picks.
"""
import asyncio
import dataclasses
import logging
import os
import typing

from agents.mcp import ToolFilterContext
from databricks_openai import AsyncDatabricksOpenAI

import genie

NONE = "none"
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "system.ai.openjev-qwen35-4b")
MIN_CONFIDENCE = float(os.environ.get("ROUTER_MIN_CONFIDENCE", "0.6"))

_log = logging.getLogger(__name__)


@dataclasses.dataclass
class RouterContext:
    """Run context for one agent turn; holds the routing decision once made. `on_decision`,
    when set, is called with the decision (None when the router was unsure or unavailable) as
    soon as it is made — the app uses it to stream the routing call into the chat."""
    client: AsyncDatabricksOpenAI
    message: str
    recent: list[str]
    on_decision: typing.Callable | None = None
    decision: dict | None = None  # {"choice", "confidence", "probabilities"}, or None
    decided: bool = False
    _lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock, repr=False)

    async def decide(self) -> dict | None:
        async with self._lock:
            if not self.decided:
                self.decision = await _route(self.client, self.message, self.recent)
                self.decided = True
                if self.on_decision:
                    try:
                        self.on_decision(self.decision)
                    except Exception:  # noqa: BLE001 - progress events are best-effort
                        _log.exception("on_decision callback failed")
        return self.decision


async def genie_tool_filter(context: ToolFilterContext, tool) -> bool:
    """Dynamic MCP tool filter: expose only the routed Genie agent's tools (all of them when the
    router is unsure; none when it says the turn needs no data)."""
    decision = await context.run_context.context.decide()
    return decision is None or decision["choice"] == context.server_name


async def _route(client: AsyncDatabricksOpenAI, message: str, recent: list[str]) -> dict | None:
    options = {name: genie.DESCRIPTIONS.get(name) for name in genie.SPACES}
    if not options:
        return None
    options[NONE] = ("LATEST_USER_MESSAGE needs no dataset query (greeting, thanks, general "
                     "knowledge).")
    # the gateway client's base_url is <host>/ai-gateway/<surface>/v1/; System One lives beside it
    url = str(client.base_url).split("/ai-gateway/")[0] + "/ai-gateway/typesafe/v1/systemone"
    body = {
        "model": ROUTER_MODEL,
        # Judge the latest message; earlier turns only disambiguate follow-ups. (Including
        # them as equals let a chit-chat turn pull data questions towards "none".)
        "state": {"LATEST_USER_MESSAGE": message, "PREVIOUS_TURNS": recent[-4:]},
        "questions": {"genie_agent": {
            "type": "choice",
            "instructions": "Decide which data source must be queried to answer "
                            "LATEST_USER_MESSAGE. Judge the latest message only; use "
                            "PREVIOUS_TURNS solely to resolve follow-ups such as 'and in 1996?' "
                            "or 'what about there?', which continue the dataset of the previous "
                            "question.",
            "criteria": options,
        }},
    }
    try:
        resp = await client.post(url, body=body, cast_to=object)
        answer = resp["answers"]["genie_agent"]
    except Exception as e:  # noqa: BLE001 - router is best-effort
        _log.warning("%s routing unavailable, exposing all Genie tools: %s", ROUTER_MODEL, e)
        return None
    decision = {"choice": answer.get("choice"), "confidence": answer.get("confidence") or 0.0,
                "probabilities": answer.get("probabilities", {})}
    _log.info("%s route: %s", ROUTER_MODEL, decision)
    if decision["choice"] not in options or decision["confidence"] < MIN_CONFIDENCE:
        return None
    return decision
