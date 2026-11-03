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

    def test_only_confirmation_starts_selected_conversation(self):
        with patch.object(self.app.dash, "ctx", SimpleNamespace(triggered_id="memory-confirm")):
            self.app._memory_progress(1, 0, [], 0)
        self.start_mock.assert_called_once_with("alice", "chat")

    def test_dash_layout_contains_button_confirmation_and_live_status(self):
        response = self.app.server.test_client().get(
            "/_dash-layout", headers={"X-Forwarded-Email": "alice"})
        self.assertEqual(response.status_code, 200)
        layout = response.get_data(as_text=True)
        for label in ["distill-memory", "Distill memory", "memory-confirm", "memory-status"]:
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


if __name__ == "__main__":
    unittest.main()
