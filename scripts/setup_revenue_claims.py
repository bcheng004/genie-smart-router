"""Create and verify an admin-only revenue view without changing the base table or grants."""
import argparse
import os
import time

from databricks.sdk import WorkspaceClient

from claims_auth import claim_token
from scripts.generate_asana_data import qualified_schema


def view_sql(catalog: str, schema: str) -> str:
    namespace = qualified_schema(catalog, schema)
    return (
        f"CREATE VIEW {namespace}.revenue_by_claim AS\n"
        f"SELECT * FROM {namespace}.revenue\n"
        "WHERE current_oauth_custom_identity_claim() = 'admin'"
    )


def execute(client, warehouse_id: str, statement: str):
    response = client.statement_execution.execute_statement(
        warehouse_id=warehouse_id, statement=statement, wait_timeout="30s")
    deadline = time.monotonic() + 180
    while response.status.state.value in ("PENDING", "RUNNING"):
        if time.monotonic() >= deadline:
            client.statement_execution.cancel_execution(response.statement_id)
            raise TimeoutError("Revenue-claims SQL verification timed out")
        time.sleep(1)
        response = client.statement_execution.get_statement(response.statement_id)
    if response.status.state.value != "SUCCEEDED":
        raise RuntimeError(f"Revenue-claims SQL failed: {response.status.state.value}")
    return response.result.data_array if response.result else []


def marketing_view_sql(catalog: str, schema: str) -> str:
    namespace = qualified_schema(catalog, schema)
    return (
        f"CREATE VIEW {namespace}.marketing_by_claim AS\n"
        "SELECT * EXCEPT (attributed_new_mrr_usd),\n"
        "CASE WHEN current_oauth_custom_identity_claim() = 'admin' "
        "THEN attributed_new_mrr_usd ELSE NULL END AS attributed_new_mrr_usd\n"
        f"FROM {namespace}.marketing"
    )


def verify_claims(owner, warehouse_id: str, namespace: str,
                  client_id: str, client_secret: str, verify_marketing: bool = False) -> None:
    expected = int(execute(owner, warehouse_id,
                           f"SELECT COUNT(*) FROM {namespace}.revenue")[0][0])
    if expected == 0:
        raise ValueError("Revenue must contain rows to prove that the admin claim works")
    if verify_marketing:
        marketing_rows, marketing_mrr = execute(owner, warehouse_id,
            f"SELECT COUNT(*), SUM(attributed_new_mrr_usd) FROM {namespace}.marketing")[0]
    for claim in ("admin", "user", None, "invalid-role"):
        token = claim_token(owner.config.host, client_id, client_secret, claim)
        caller = WorkspaceClient(host=owner.config.host, token=token, auth_type="pat")
        actual_claim, count = execute(caller, warehouse_id,
            "SELECT current_oauth_custom_identity_claim(), COUNT(*) "
            f"FROM {namespace}.revenue_by_claim")[0]
        allowed = expected if claim == "admin" else 0
        if actual_claim != claim or int(count) != allowed:
            raise RuntimeError(f"Revenue-claims verification failed for {claim!r}")
        print(f"Verified claim {claim!r}: {count} revenue rows")
        if verify_marketing:
            actual_rows, actual_mrr, populated = execute(caller, warehouse_id,
                "SELECT COUNT(*), SUM(attributed_new_mrr_usd), COUNT(attributed_new_mrr_usd) "
                f"FROM {namespace}.marketing_by_claim")[0]
            if int(actual_rows) != int(marketing_rows):
                raise RuntimeError("Marketing row count changed under a custom claim")
            if claim == "admin":
                if actual_mrr != marketing_mrr:
                    raise RuntimeError("Admin marketing MRR does not match the source")
            elif actual_mrr is not None or int(populated) != 0:
                raise RuntimeError("Marketing exposed revenue-derived MRR without an admin claim")
            print(f"Verified claim {claim!r}: marketing revenue-derived MRR "
                  + ("visible" if claim == "admin" else "masked"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--warehouse-id", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--apply", action="store_true", help="Create a new view; never replace an existing object")
    parser.add_argument("--marketing", action="store_true", help="Create the marketing view with revenue-derived MRR masked")
    parser.add_argument("--verify-claims", action="store_true",
                        help="Verify admin, user, missing and unknown OAuth claims on the existing view")
    args = parser.parse_args()
    namespace = qualified_schema(args.catalog, args.schema)
    statement = (marketing_view_sql if args.marketing else view_sql)(args.catalog, args.schema)
    client_id = os.environ.get("REVENUE_CLAIMS_CLIENT_ID", "")
    client_secret = os.environ.get("REVENUE_CLAIMS_CLIENT_SECRET", "")
    if args.verify_claims and not (client_id and client_secret):
        parser.error("--verify-claims requires REVENUE_CLAIMS_CLIENT_ID and REVENUE_CLAIMS_CLIENT_SECRET")
    if not args.apply and not args.verify_claims:
        print(statement + ";")
        return
    owner = WorkspaceClient(profile=args.profile)
    if args.apply:
        execute(owner, args.warehouse_id, statement)
        view = "marketing_by_claim" if args.marketing else "revenue_by_claim"
        print(f"Created {namespace}.{view}")
    if args.apply and not args.marketing:
        actual_claim, count = execute(owner, args.warehouse_id,
            "SELECT current_oauth_custom_identity_claim(), COUNT(*) "
            f"FROM {namespace}.revenue_by_claim")[0]
        if actual_claim is None and int(count) != 0:
            raise RuntimeError("A token without a claim unexpectedly returned revenue rows")
        if actual_claim is None:
            print("Verified current token has no claim: 0 revenue rows")
    if args.verify_claims:
        verify_claims(owner, args.warehouse_id, namespace, client_id, client_secret,
                      verify_marketing=args.marketing)
    print("This view does not revoke direct table access or change the app's authentication.")


if __name__ == "__main__":
    main()
