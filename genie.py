"""Genie tools using user-scoped MCP or server-assigned OAuth claims.

With GENIE_CUSTOM_CLAIMS=true, membership tools use the Genie Conversation API with
claim-bearing service-principal tokens. Otherwise they retain the on-behalf-of flow below.

Each Genie agent is a Databricks managed MCP server at
https://<workspace-host>/api/2.0/mcp/genie/{genie_space_id}.

On-behalf-of auth mirrors https://github.com/bcheng004/databricks-agent-on-apps-mcp-blog
(agent_server/utils.py + multi_server_mcp_client.py):
  * the user's client is `WorkspaceClient(token=<x-forwarded-access-token>, auth_type="pat")`
    — Databricks Apps forwards that token with the app's user API scopes; Genie MCP requires
    the `genie` scope (users who signed in before it was added must sign in again);
  * MCP requests carry `workspace_client.config.authenticate()` headers directly, instead of
    the databricks_mcp OAuth client provider (which 403s with a forwarded user token).
So Genie runs every query as the user, with their permissions. There is deliberately no
fallback to the app's service principal: without a user token the Genie tools aren't offered.
"""
import os
import asyncio
import json
from datetime import timedelta

from agents import function_tool
from agents.mcp import MCPServerStreamableHttp
from databricks.sdk import WorkspaceClient
from contextlib import AsyncExitStack, asynccontextmanager

import claims_auth

# name -> space id, from the genie_space app resources in databricks.yml (value_from)
DATASET = os.environ.get("GENIE_DATASET", "samples")
SPACE_VARIABLES = ({
    "genie_asana_revenue": "ASANA_REVENUE_GENIE_SPACE_ID",
    "genie_asana_marketing": "ASANA_MARKETING_GENIE_SPACE_ID",
} if DATASET == "asana" else {
    "genie_tpch_orders": "TPCH_GENIE_SPACE_ID",
    "genie_nyc_taxi": "NYC_TAXI_GENIE_SPACE_ID",
})
SPACES = {name: os.environ[variable] for name, variable in SPACE_VARIABLES.items()
          if os.environ.get(variable)}
CHAT_TOPICS = ("membership revenue or marketing campaigns" if DATASET == "asana"
               else "TPC-H orders or NYC taxi trips")

# Display names for the chat's route/tool boxes (internal keys stay wired to env vars)
LABELS = {
    "genie_asana_revenue": "membership_revenue",
    "genie_asana_marketing": "marketing_campaigns",
    "genie_tpch_orders": "tpch_orders",
    "genie_nyc_taxi": "nyc_taxi",
}


def label(name: str | None) -> str:
    """User-facing name of a Genie agent/tool (falls back to the key sans `genie_`)."""
    return LABELS.get(name or "", (name or "?").removeprefix("genie_"))


# What each Genie agent covers (the router model picks one per turn from these)
DESCRIPTIONS = {
    "genie_asana_revenue": "Paid-membership revenue: monthly membership revenue, annual membership "
                          "revenue run-rate, members, membership tiers, member households, regions, "
                          "expansion and churn.",
    "genie_asana_marketing": "Membership marketing campaigns: campaign spend, channel performance, "
                            "leads, conversion funnel, acquisition costs and attributed new membership "
                            "revenue. Includes a paid-social lead-quality decline.",
    "genie_tpch_orders": "TPC-H B2B orders data: orders, revenue, customers, market segments, "
                         "nations/regions, line items, shipping modes and returns (1992-1998).",
    "genie_nyc_taxi": "NYC yellow taxi trips (Jan-Feb 2016): trip counts, fares, distances, "
                      "durations, pickup/dropoff ZIP codes.",
}


def user_client(user_token: str | None) -> WorkspaceClient | None:
    """WorkspaceClient acting as the end user (on-behalf-of), or None without a token.
    Locally (not a Databricks App), your CLI profile stands in for the user."""
    if not os.environ.get("DATABRICKS_APP_NAME"):
        profile = os.environ.get("DATABRICKS_CONFIG_PROFILE")
        return WorkspaceClient(profile=profile) if profile else None
    return WorkspaceClient(token=user_token, auth_type="pat") if user_token else None


def query_with_claim(client, space_id: str, question: str) -> str:
    """Use a new conversation per invocation; no shared-principal conversation IDs are accepted."""
    message = client.genie.start_conversation_and_wait(
        space_id, question, timeout=timedelta(minutes=5))
    output = []
    for attachment in message.attachments or []:
        if attachment.text and attachment.text.content:
            output.append({"text": attachment.text.content})
        if attachment.query:
            result = client.genie.get_message_attachment_query_result(
                space_id, message.conversation_id, message.id, attachment.attachment_id)
            statement = result.statement_response
            if not statement or not statement.status or statement.status.state.value != "SUCCEEDED":
                raise RuntimeError("Claim-aware Genie query did not complete successfully")
            result_data = statement.as_dict()
            if result_data.get("result", {}).get("data_array"):
                rows = result_data["result"]["data_array"]
                result_data["result"]["data_array"] = rows[:100]
                result_data["truncated_for_chat"] = len(rows) > 100
            output.append({"sql": attachment.query.query, "result": result_data})
    return json.dumps(output)


def _claim_tool(name: str, space_id: str, client, role: str):
    async def allowed(context, agent):
        decision = await context.context.decide()
        return decision is None or decision["choice"] == name

    @function_tool(name_override=name, description_override=DESCRIPTIONS[name], is_enabled=allowed)
    async def query(question: str) -> str:
        """Ask a self-contained question about the retailer's membership data."""
        if name == "genie_asana_revenue" and role != "admin":
            return "Access denied: the user role cannot read revenue. Do not infer or invent revenue values."
        return await asyncio.to_thread(query_with_claim, client, space_id, question)

    return query


async def claim_tools(end_user_client, email: str, requested_role: str | None = None):
    if DATASET != "asana":
        raise ValueError("Custom revenue claims require the membership dataset")
    client = await asyncio.to_thread(claims_auth.client_for, end_user_client, email, requested_role)
    return [_claim_tool(name, space_id, client, claims_auth.effective_role(email, requested_role))
            for name, space_id in SPACES.items()]


@asynccontextmanager
async def mcp_servers(client: WorkspaceClient | None, tool_filter=None):
    """Connected Genie MCP servers for one agent run, authenticated with `client`. Each is
    skipped when its client is None. `tool_filter` is an Agents SDK MCP tool filter (e.g. the
    router's) applied to each."""
    async with AsyncExitStack() as stack:
        servers = []
        for name, space_id in SPACES.items():
            if client is None:
                continue
            servers.append(await stack.enter_async_context(MCPServerStreamableHttp(
                name=name,
                params={"url": f"{client.config.host.rstrip('/')}/api/2.0/mcp/genie/{space_id}",
                        # fresh credentials from the client (one run = one request)
                        "headers": dict(client.config.authenticate()),
                        "timeout": 30, "sse_read_timeout": 600},
                client_session_timeout_seconds=300,  # Genie queries can take a while
                cache_tools_list=True, tool_filter=tool_filter)))
        yield servers
