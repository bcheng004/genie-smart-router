"""Shared server-assigned revenue roles and OAuth custom-claim authentication."""
import base64
import json
import os
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from databricks.sdk import WorkspaceClient

ADMIN_EMAIL = "bo.cheng@databricks.com"
MEMORY_POLICY = "revenue-claims-v1"


def enabled() -> bool:
    return os.environ.get("GENIE_CUSTOM_CLAIMS", "false").lower() == "true"


def canonical_email(email: str | None) -> str:
    return (email or "").strip().casefold()


def role_for(email: str | None) -> str:
    return "admin" if canonical_email(email) == ADMIN_EMAIL else "user"


def effective_role(email: str | None, requested_role: str | None = None) -> str:
    if role_for(email) != "admin":
        return "user"
    return "admin" if requested_role is None or requested_role == "admin" else "user"


def memory_actor(email: str, requested_role: str | None = None) -> str:
    if not enabled():
        return email
    return f"{canonical_email(email)}|{MEMORY_POLICY}|{effective_role(email, requested_role)}"


def claim_token(host: str, client_id: str, client_secret: str, claim: str | None) -> str:
    """Mint a token on the trusted server; never expose this as a user role picker."""
    parsed = urlsplit(host)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("OAuth requires an HTTPS workspace host without embedded credentials")
    if not client_id or not client_secret:
        raise ValueError("A service-principal client ID and OAuth secret are required")
    fields = {"grant_type": "client_credentials", "scope": "all-apis"}
    if claim is not None:
        fields["custom_claim"] = claim
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    request = Request(
        f"{host.rstrip('/')}/oidc/v1/token", data=urlencode(fields).encode(),
        headers={"Authorization": f"Basic {credentials}",
                 "Content-Type": "application/x-www-form-urlencoded"}, method="POST")
    with urlopen(request, timeout=30) as response:
        token = json.load(response).get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("OAuth response did not contain an access token")
    return token


def client_for(end_user_client, email: str, requested_role: str | None = None):
    """Validate the end user before selecting a server-owned role and issuing credentials."""
    if end_user_client is None or not canonical_email(email):
        raise PermissionError("A signed-in user is required for claim-aware Genie access")
    actual = canonical_email(end_user_client.current_user.me().user_name)
    if not actual or actual != canonical_email(email):
        raise PermissionError("The signed-in identity does not match the requested user")
    prefix = "DATABRICKS" if os.environ.get("DATABRICKS_APP_NAME") else "REVENUE_CLAIMS"
    token = claim_token(end_user_client.config.host,
                        os.environ.get(f"{prefix}_CLIENT_ID", ""),
                        os.environ.get(f"{prefix}_CLIENT_SECRET", ""),
                        effective_role(actual, requested_role))
    return WorkspaceClient(host=end_user_client.config.host, token=token, auth_type="pat")
