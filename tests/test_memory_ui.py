import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class MemoryUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {
            "DATABRICKS_HOST": "https://offline.invalid", "DATABRICKS_TOKEN": "offline-test-token",
            "DATABRICKS_AUTH_TYPE": "pat", "LONG_TERM_MEMORY_VOLUME": "",
        }), patch("agents.set_default_openai_client"), patch("agents.set_default_openai_api"), \
                patch("databricks_openai.utils.clients.WorkspaceClient") as workspace, \
                patch("threading.Thread.start"):
            workspace.return_value.config.host = "https://offline.invalid"
            cls.app = importlib.import_module("app")

    def setUp(self):
        self.user = patch.object(self.app, "_user", return_value="alice")
        self.user.start()
        self.addCleanup(self.user.stop)
        self.configured = patch("memory_actions.configured", return_value=True)
        self.configured.start()
        self.addCleanup(self.configured.stop)
        self.start = patch("memory_actions.start")
        self.start_mock = self.start.start()
        self.addCleanup(self.start.stop)
        self.state = patch.dict(self.app._user_data, {"alice": {
            "active": "chat", "sessions": [], "router": "task_v2"}}, clear=True)
        self.state.start()
        self.addCleanup(self.state.stop)
        self.live = patch.dict(self.app._live, {}, clear=True)
        self.live.start()
        self.addCleanup(self.live.stop)
        self.app.memory_actions._jobs.clear()
        for target, value in [("long_term_memory.source_path", "/Volumes/demo/memory/events"),
                              ("memory.use_durable", True)]:
            configured = patch(target, return_value=value)
            configured.start()
            self.addCleanup(configured.stop)
        self.listing = patch("long_term_memory.list_memories", return_value=[])
        self.list_mock = self.listing.start()
        self.addCleanup(self.listing.stop)

    def test_deployed_identity_never_falls_back_to_dev_admin(self):
        with patch.dict(os.environ, {"DATABRICKS_APP_NAME": "router",
                                    "DEV_USER": "bo.cheng@databricks.com"}), \
                self.app.app.server.test_request_context("/"):
            self.assertEqual(self.app._user_email(), "local-user")
        with self.app.app.server.test_request_context(
                "/", headers={"X-Forwarded-Email": "bo.cheng@databricks.com"}):
            self.assertEqual(self.app._user_email(), "bo.cheng@databricks.com")

    def test_claim_dropdown_is_admin_only_and_user_selection_is_locked(self):
        def find_component(node, component_id):
            if isinstance(node, dict):
                if node.get("props", {}).get("id") == component_id:
                    return node["props"]
                for value in node.values():
                    found = find_component(value, component_id)
                    if found:
                        return found
            elif isinstance(node, list):
                for value in node:
                    found = find_component(value, component_id)
                    if found:
                        return found

        for email, disabled, choices in (("bo.cheng@databricks.com", False, ["admin", "user"]),
                                         ("alice", True, ["user"])):
            response = self.app.server.test_client().get(
                "/_dash-layout", headers={"X-Forwarded-Email": email})
            self.assertEqual(response.status_code, 200)
            dropdown = find_component(response.get_json(), "claim-role")
            self.assertEqual(dropdown["disabled"], disabled)
            self.assertEqual([choice["value"] for choice in dropdown["options"]], choices)

    def test_claim_switch_clears_admin_transcript_and_restores_it_only_in_admin_mode(self):
        import json
        import claims_auth

        self.user.stop()
        admin = "bo.cheng@databricks.com|revenue-claims-v1|admin"
        user = "bo.cheng@databricks.com|revenue-claims-v1|user"
        self.app._user_data[admin] = {"active": "admin-chat", "router": "task_v2", "sessions": [
            {"id": "admin-chat", "title": "Admin-only revenue", "created": 1, "updated": 1,
             "events": [{"kind": "reply", "text": "ADMIN-ONLY-REVENUE"}], "branch_of": None}]}
        with patch.dict(os.environ, {"GENIE_CUSTOM_CLAIMS": "true"}), \
                patch("memory.use_durable", return_value=False), \
                self.app.server.test_request_context("/", headers={"X-Forwarded-Email": claims_auth.ADMIN_EMAIL}), \
                patch.object(self.app.dash, "ctx", SimpleNamespace(triggered_id="claim-role")):
            response = self.app.on_message(0, 0, 0, 0, [], [], "task_v2", "user", "unused")
            self.assertNotIn("ADMIN-ONLY-REVENUE", json.dumps(response, default=str))
            self.assertEqual(response[2:], ("", False, False, True))
            self.assertIn(user, self.app._user_data)
            response = self.app.on_message(0, 0, 0, 0, [], [], "task_v2", "admin", "unused")
            self.assertIn("ADMIN-ONLY-REVENUE", json.dumps(response, default=str))

    def test_claim_change_refreshes_selected_role_memories_even_when_idle(self):
        with patch.object(self.app.dash, "ctx", SimpleNamespace(triggered_id="claim-role")):
            self.app._refresh_long_term_memory(0, True, "user", False)
        self.app._user.assert_called_once_with("user")
        self.list_mock.assert_called_once_with("alice")

    def test_claim_change_dismisses_pending_memory_confirmation(self):
        with patch.object(self.app.dash, "ctx", SimpleNamespace(triggered_id="claim-role")):
            self.assertFalse(self.app._confirm_distillation(1, "user"))
        self.start_mock.assert_not_called()

    def test_claim_change_takes_priority_over_coalesced_idle_poll(self):
        with patch.object(self.app.dash, "ctx", SimpleNamespace(
                triggered_id="memory-tick", triggered_prop_ids={
                    "memory-tick.disabled": "memory-tick", "claim-role.value": "claim-role"})):
            self.app._refresh_long_term_memory(0, True, "user", False)
        self.list_mock.assert_called_once_with("alice")

    def test_only_confirmation_starts_selected_conversation(self):
        with patch.object(self.app.dash, "ctx", SimpleNamespace(triggered_id="memory-confirm")):
            self.app._memory_progress(1, 0, [], 0)
        self.start_mock.assert_called_once_with("alice", "chat")

    def test_dash_layout_contains_button_confirmation_and_live_status(self):
        response = self.app.server.test_client().get(
            "/_dash-layout", headers={"X-Forwarded-Email": "alice"})
        self.assertEqual(response.status_code, 200)
        layout = response.get_data(as_text=True)
        for label in ["distill-memory", "Distill memory", "memory-confirm", "memory-status",
                      "Long-term memory", "long-term-memory-list", "refresh-long-term-memory"]:
            self.assertIn(label, layout)
        self.start_mock.assert_not_called()

    def test_status_poll_does_not_replay_a_previous_confirmation(self):
        with patch.object(self.app.dash, "ctx", SimpleNamespace(triggered_id="memory-tick")):
            self.app._memory_progress(1, 1, [], 0)
        self.start_mock.assert_not_called()

    def test_busy_chat_cannot_export_an_unsettled_turn(self):
        self.app._live["alice"] = {"done": False}
        with patch.object(self.app.dash, "ctx", SimpleNamespace(triggered_id="memory-confirm")):
            message, disabled, _ = self.app._memory_progress(1, 0, [], 0)
        self.start_mock.assert_not_called()
        self.assertTrue(disabled)
        self.assertIn("finish", message)

    def test_no_selected_chat_disables_distillation(self):
        self.app._user_data["alice"]["active"] = None
        with patch.object(self.app.dash, "ctx", SimpleNamespace(triggered_id=None)):
            _, disabled, _ = self.app._memory_progress(0, 0, [], 0)
        self.assertTrue(disabled)

    def test_initial_memory_load_is_scoped_to_signed_in_actor(self):
        with patch.object(self.app.dash, "ctx", SimpleNamespace(triggered_id=None)):
            body, running = self.app._refresh_long_term_memory(0, True, None, False)
        self.list_mock.assert_called_once_with("alice")
        self.assertFalse(running)
        self.assertIn("No saved facts yet", body.children)

    def test_distillation_completion_refreshes_the_panel(self):
        with patch.object(self.app.dash, "ctx", SimpleNamespace(triggered_id="memory-tick")):
            self.app._refresh_long_term_memory(0, True, None, True)
        self.list_mock.assert_called_once_with("alice")

    def test_initial_upstream_interval_update_still_loads_memories(self):
        with patch.object(self.app.dash, "ctx", SimpleNamespace(triggered_id="memory-tick")):
            body, running = self.app._refresh_long_term_memory(0, True, None, None)
        self.list_mock.assert_called_once_with("alice")
        self.assertFalse(running)
        self.assertIn("No saved facts yet", body.children)

    def test_active_or_unchanged_idle_poll_does_not_query_storage(self):
        for disabled, was_running in [(False, False), (False, True), (True, False)]:
            with patch.object(self.app.dash, "ctx", SimpleNamespace(triggered_id="memory-tick")):
                body, running = self.app._refresh_long_term_memory(0, disabled, None, was_running)
            self.assertIs(body, self.app.dash.no_update)
            self.assertEqual(running, not disabled)
        self.list_mock.assert_not_called()

    def test_manual_refresh_loads_current_facts_as_plain_text(self):
        self.list_mock.return_value = [{"key": "preferences.currency", "content": "Use <USD>.",
                                        "version": 2, "source_at": 100.0}]
        with patch.object(self.app.dash, "ctx", SimpleNamespace(triggered_id="refresh-long-term-memory")):
            cards, _ = self.app._refresh_long_term_memory(1, False, None, True)
        self.assertEqual(cards[0].children[1].children, "Use <USD>.")
        self.assertIn("Version 2", cards[0].children[2].children)
        self.list_mock.assert_called_once_with("alice")

    def test_storage_error_has_retry_message_without_backend_details(self):
        self.list_mock.side_effect = RuntimeError("private backend detail")
        body = self.app._long_term_memory_view("alice")
        self.assertEqual(body.role, "alert")
        self.assertIn("Refresh", body.children)
        self.assertNotIn("private backend detail", body.children)

    def test_unconfigured_storage_is_not_reported_as_empty_memory(self):
        with patch("memory.use_durable", return_value=False):
            body = self.app._long_term_memory_view("alice")
        self.assertIn("not available", body.children)
        self.list_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
