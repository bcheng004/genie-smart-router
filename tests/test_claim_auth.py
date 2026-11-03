import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import claims_auth
import genie
from agents.tool_context import ToolContext
from scripts import create_genie_agents
from scripts.asana_agents import build_asana_agents


class ClaimAuthTests(unittest.TestCase):
    def test_only_bo_is_admin(self):
        self.assertEqual(claims_auth.role_for("bo.cheng@databricks.com"), "admin")
        self.assertEqual(claims_auth.role_for(" BO.CHENG@DATABRICKS.COM "), "admin")
        for email in (None, "", "other@databricks.com", "bo.cheng@databricks.com.attacker.com",
                      "bo.cheng+admin@databricks.com", "admin"):
            with self.subTest(email=email):
                self.assertEqual(claims_auth.role_for(email), "user")

    def test_memory_scope_does_not_reuse_legacy_or_other_users_facts(self):
        with patch.dict(os.environ, {"GENIE_CUSTOM_CLAIMS": "true"}):
            admin = claims_auth.memory_actor(claims_auth.ADMIN_EMAIL)
            user = claims_auth.memory_actor("other@databricks.com")
            self.assertEqual(admin, "bo.cheng@databricks.com|revenue-claims-v1|admin")
            self.assertEqual(user, "other@databricks.com|revenue-claims-v1|user")
            self.assertNotEqual(admin, user)
            self.assertNotEqual(admin, claims_auth.ADMIN_EMAIL)
        with patch.dict(os.environ, {"GENIE_CUSTOM_CLAIMS": "false"}):
            self.assertEqual(claims_auth.memory_actor("alice"), "alice")

    def test_mismatched_or_missing_verified_identity_cannot_mint_a_claim(self):
        caller = Mock()
        caller.current_user.me.return_value.user_name = "other@databricks.com"
        for client, email in ((None, claims_auth.ADMIN_EMAIL), (caller, None),
                              (caller, claims_auth.ADMIN_EMAIL)):
            with self.subTest(email=email), patch.object(claims_auth, "claim_token") as mint:
                with self.assertRaises(PermissionError):
                    claims_auth.client_for(client, email)
                mint.assert_not_called()

    def test_verified_identity_determines_claim_and_app_credentials_stay_server_side(self):
        for email, role in ((claims_auth.ADMIN_EMAIL, "admin"), ("other@databricks.com", "user")):
            caller = Mock()
            caller.config.host = "https://workspace.example"
            caller.current_user.me.return_value.user_name = email
            with patch.dict(os.environ, {"DATABRICKS_APP_NAME": "router",
                                        "DATABRICKS_CLIENT_ID": "app-id",
                                        "DATABRICKS_CLIENT_SECRET": "app-secret"}), \
                    patch.object(claims_auth, "claim_token", return_value="scoped-token") as mint, \
                    patch.object(claims_auth, "WorkspaceClient") as workspace:
                claims_auth.client_for(caller, email)
                mint.assert_called_once_with("https://workspace.example", "app-id", "app-secret", role)
                workspace.assert_called_once_with(host="https://workspace.example",
                                                  token="scoped-token", auth_type="pat")

    def test_token_errors_do_not_fall_back_to_unclaimed_credentials(self):
        caller = Mock()
        caller.current_user.me.return_value.user_name = claims_auth.ADMIN_EMAIL
        with patch.object(claims_auth, "claim_token", side_effect=RuntimeError("OAuth failed")), \
                patch.object(claims_auth, "WorkspaceClient") as workspace:
            with self.assertRaises(RuntimeError):
                claims_auth.client_for(caller, claims_auth.ADMIN_EMAIL)
            workspace.assert_not_called()

    def test_claim_spaces_and_examples_reference_only_views(self):
        agents = build_asana_agents("demo", "asana", create_genie_agents._col, claim_views=True)
        for agent, view in zip(agents, ("revenue_by_claim", "marketing_by_claim")):
            self.assertEqual(list(agent["tables"]), [f"demo.asana.{view}"])
            self.assertIn("Claims", agent["title"])
            for _, sql in agent["examples"]:
                self.assertIn(f"demo.asana.{view}", sql)
                self.assertNotIn(f"demo.asana.{view.removesuffix('_by_claim')} AS", sql)


class ClaimGenieTests(unittest.IsolatedAsyncioTestCase):
    async def test_user_revenue_tool_is_denied_without_sending_a_question(self):
        client = Mock()
        tool = genie._claim_tool("genie_asana_revenue", "revenue-space", client, "user")
        arguments = json.dumps({"question": "Show all revenue"})
        context = ToolContext(context=None, tool_name=tool.name, tool_call_id="test-call",
                              tool_arguments=arguments)
        result = await tool.on_invoke_tool(context, arguments)
        self.assertIn("Access denied", result)
        client.genie.start_conversation_and_wait.assert_not_called()

    async def test_claim_tools_preserve_router_choices_and_uncertain_fallback(self):
        tool = genie._claim_tool("genie_asana_marketing", "space", Mock(), "user")
        for decision, expected in ((None, True), ({"choice": "genie_asana_marketing"}, True),
                                   ({"choice": "genie_asana_revenue"}, False),
                                   ({"choice": "none"}, False)):
            async def decide():
                return decision
            context = SimpleNamespace(context=SimpleNamespace(decide=decide))
            self.assertEqual(await tool.is_enabled(context, None), expected)

    async def test_claim_authentication_is_required_before_any_tool_is_exposed(self):
        with patch.object(genie, "DATASET", "asana"), \
                patch.object(claims_auth, "client_for", side_effect=PermissionError("No identity")):
            with self.assertRaises(PermissionError):
                await genie.claim_tools(None, claims_auth.ADMIN_EMAIL)

    def test_each_question_starts_a_new_conversation(self):
        client = Mock()
        client.genie.start_conversation_and_wait.return_value = SimpleNamespace(attachments=[])
        for _ in range(2):
            self.assertEqual(genie.query_with_claim(client, "space", "Question"), "[]")
        self.assertEqual(client.genie.start_conversation_and_wait.call_count, 2)
        client.genie.create_message.assert_not_called()

    def test_genie_query_result_is_checked_and_limited(self):
        client = Mock()
        message = SimpleNamespace(conversation_id="owned-conversation", id="owned-message",
            attachments=[SimpleNamespace(text=None, attachment_id="result", query=SimpleNamespace(
                query="SELECT * FROM demo.asana.marketing_by_claim"))])
        client.genie.start_conversation_and_wait.return_value = message
        statement = Mock()
        statement.status.state.value = "SUCCEEDED"
        statement.as_dict.return_value = {"result": {"data_array": [[index] for index in range(110)]}}
        client.genie.get_message_attachment_query_result.return_value.statement_response = statement
        output = json.loads(genie.query_with_claim(client, "space", "Question"))
        self.assertEqual(len(output[0]["result"]["result"]["data_array"]), 100)
        self.assertTrue(output[0]["result"]["truncated_for_chat"])
        client.genie.get_message_attachment_query_result.assert_called_once_with(
            "space", "owned-conversation", "owned-message", "result")
        statement.status.state.value = "FAILED"
        with self.assertRaises(RuntimeError):
            genie.query_with_claim(client, "space", "Question")


if __name__ == "__main__":
    unittest.main()
