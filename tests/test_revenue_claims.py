import io
import json
import sqlite3
import unittest
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import parse_qs

from scripts import setup_revenue_claims as claims
import claims_auth


class RevenueClaimsTests(unittest.TestCase):
    def test_view_is_in_same_namespace_and_does_not_change_base_table(self):
        self.assertEqual(claims.view_sql("workshop_guy_catalog", "asana_demo"),
                         "CREATE VIEW workshop_guy_catalog.asana_demo.revenue_by_claim AS\n"
                         "SELECT * FROM workshop_guy_catalog.asana_demo.revenue\n"
                         "WHERE current_oauth_custom_identity_claim() = 'admin'")
        with self.assertRaises(ValueError):
            claims.view_sql("demo; DROP TABLE revenue", "asana_demo")

    def test_sql_predicate_denies_everything_except_exact_admin_claim(self):
        with closing(sqlite3.connect(":memory:")) as database:
            database.execute("CREATE TABLE revenue (amount INTEGER)")
            database.executemany("INSERT INTO revenue VALUES (?)", [(100,), (200,)])
            statement = claims.view_sql("demo", "asana").replace("demo.asana.", "")
            database.execute(statement)
            for claim, expected in [("admin", 2), ("user", 0), (None, 0), ("", 0),
                                    ("Admin", 0), ("invalid-role", 0), ("admin,user", 0)]:
                with self.subTest(claim=claim):
                    database.create_function("current_oauth_custom_identity_claim", 0,
                                             lambda: claim)
                    self.assertEqual(database.execute(
                        "SELECT COUNT(*) FROM revenue_by_claim").fetchone()[0], expected)

    def test_token_request_carries_custom_claim_and_omits_missing_claim(self):
        for claim in ("admin", "user", None):
            with self.subTest(claim=claim), patch.object(claims_auth, "urlopen") as send:
                send.return_value.__enter__.return_value = io.BytesIO(
                    json.dumps({"access_token": "test-token"}).encode())
                self.assertEqual(claims.claim_token("https://workspace.example", "client",
                                                    "secret", claim), "test-token")
                request = send.call_args.args[0]
                fields = parse_qs(request.data.decode())
                self.assertEqual(request.full_url, "https://workspace.example/oidc/v1/token")
                self.assertEqual(fields["grant_type"], ["client_credentials"])
                self.assertEqual(fields["scope"], ["all-apis"])
                self.assertEqual(fields.get("custom_claim"), [claim] if claim else None)
                self.assertNotIn("secret", request.data.decode())

    def test_credentials_require_https(self):
        for host in ("http://workspace.example", "https://user:password@workspace.example"):
            with self.assertRaises(ValueError), patch.object(claims_auth, "urlopen") as send:
                claims.claim_token(host, "client", "secret", "admin")
            send.assert_not_called()

    def test_verification_checks_claim_identity_as_well_as_row_count(self):
        owner = SimpleNamespace(config=SimpleNamespace(host="https://workspace.example"))
        results = [[[2]], [["admin", "2"]], [["user", "0"]], [[None, "0"]],
                   [["invalid-role", "0"]]]
        with patch.object(claims, "execute", side_effect=results), \
                patch.object(claims, "claim_token", return_value="token") as mint, \
                patch.object(claims, "WorkspaceClient"), patch("builtins.print"):
            claims.verify_claims(owner, "warehouse", "demo.asana", "client", "secret")
        self.assertEqual([call.args[-1] for call in mint.call_args_list],
                         ["admin", "user", None, "invalid-role"])
        with patch.object(claims, "execute", side_effect=[[[2]], [[None, "2"]]]), \
                patch.object(claims, "claim_token", return_value="token"), \
                patch.object(claims, "WorkspaceClient"):
            with self.assertRaises(RuntimeError):
                claims.verify_claims(owner, "warehouse", "demo.asana", "client", "secret")

    def test_sql_polls_and_propagates_execution_failure(self):
        client = Mock()
        pending = SimpleNamespace(statement_id="statement", status=SimpleNamespace(
            state=SimpleNamespace(value="PENDING")))
        complete = SimpleNamespace(status=SimpleNamespace(state=SimpleNamespace(value="SUCCEEDED")),
                                   result=SimpleNamespace(data_array=[[0]]))
        client.statement_execution.execute_statement.return_value = pending
        client.statement_execution.get_statement.return_value = complete
        with patch.object(claims.time, "sleep"):
            self.assertEqual(claims.execute(client, "warehouse", "SELECT 0"), [[0]])
        client.statement_execution.get_statement.assert_called_once_with("statement")
        complete.status.state.value = "FAILED"
        with patch.object(claims.time, "sleep"), self.assertRaises(RuntimeError):
            claims.execute(client, "warehouse", "SELECT 0")


if __name__ == "__main__":
    unittest.main()
