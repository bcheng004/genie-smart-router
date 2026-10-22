"""Genie agents as MCP tools, called on behalf of the signed-in user.

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

from agents.mcp import MCPServerStreamableHttp
from databricks.sdk import WorkspaceClient
from contextlib import AsyncExitStack, asynccontextmanager

# name -> space id, from the genie_space app resources in databricks.yml (value_from)
SPACES = {name: sid for name, sid in {
    "genie_tpch_orders": os.environ.get("TPCH_GENIE_SPACE_ID"),
    "genie_nyc_taxi": os.environ.get("NYC_TAXI_GENIE_SPACE_ID"),
}.items() if sid}

# What each Genie agent covers (the router model picks one per turn from these)
DESCRIPTIONS = {
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
