import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import memory_actions


class MemoryActionsTests(unittest.TestCase):
    def setUp(self):
        memory_actions._jobs.clear()
        self.environment = patch.dict(os.environ, {"LONG_TERM_MEMORY_PIPELINE_ID": "pipeline"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.client = MagicMock()
        self.client.config.host = "https://workspace.example"
        self.client.pipelines.get.return_value.latest_updates = []
        self.client.pipelines.start_update.return_value.update_id = "update"
        self.client.pipelines.get_update.return_value.update.state = "COMPLETED"
        self.workspace = patch("memory._workspace_client", return_value=self.client)
        self.workspace.start()
        self.addCleanup(self.workspace.stop)
        self.queue = patch("long_term_memory.queue_chat", return_value=1)
        self.queue_mock = self.queue.start()
        self.addCleanup(self.queue.stop)
        self.publish = patch("long_term_memory.publish_pending", side_effect=[1, 0])
        self.publish_mock = self.publish.start()
        self.addCleanup(self.publish.stop)

    def test_manual_run_exports_only_authenticated_actors_selected_chat(self):
        memory_actions._run("alice", "chat")
        self.queue_mock.assert_called_once_with("alice", "chat")
        self.assertEqual(self.publish_mock.call_args.kwargs,
                         {"actor_id": "alice", "chat_id": "chat"})
        self.client.pipelines.start_update.assert_called_once_with("pipeline")
        self.assertIn("Distillation complete", memory_actions.status("alice")["message"])
        self.assertFalse(memory_actions.status("alice")["running"])
        self.assertNotIn("url", memory_actions.status("bob"))

    def test_active_pipeline_does_not_export_or_start_another_update(self):
        self.client.pipelines.get.return_value.latest_updates = [SimpleNamespace(state="RUNNING")]
        memory_actions._run("alice", "chat")
        self.queue_mock.assert_not_called()
        self.publish_mock.assert_not_called()
        self.client.pipelines.start_update.assert_not_called()
        self.assertIn("already busy", memory_actions.status("alice")["message"])

    def test_failure_exposes_status_and_link_not_a_false_success(self):
        self.client.pipelines.get_update.return_value.update.state = "FAILED"
        memory_actions._run("alice", "chat")
        progress = memory_actions.status("alice")
        self.assertFalse(progress["running"])
        self.assertIn("failed", progress["message"])
        self.assertTrue(progress["url"].endswith("/pipelines/pipeline/updates/update"))

    def test_upload_failure_never_starts_pipeline(self):
        self.publish_mock.side_effect = RuntimeError("upload failed")
        with self.assertLogs("memory_actions", level="ERROR"):
            memory_actions._run("alice", "chat")
        self.client.pipelines.start_update.assert_not_called()
        self.assertFalse(memory_actions.status("alice")["running"])
        self.assertIn("Could not distill", memory_actions.status("alice")["message"])

    def test_owner_check_failure_is_actionable(self):
        self.queue_mock.side_effect = ValueError("Conversation not found for the signed-in user")
        memory_actions._run("alice", "bobs-chat")
        self.publish_mock.assert_not_called()
        self.assertIn("not found", memory_actions.status("alice")["message"])

    def test_timeout_retains_update_link_without_claiming_pipeline_stopped(self):
        with patch("memory_actions.time.monotonic", side_effect=[0, 1801]):
            memory_actions._run("alice", "chat")
        self.assertIn("may still be running", memory_actions.status("alice")["message"])
        self.assertFalse(memory_actions.status("alice")["running"])

    def test_start_is_nonblocking_and_duplicate_clicks_do_not_start_workers(self):
        with patch("memory_actions.configured", return_value=True), \
                patch("memory_actions.threading.Thread") as thread:
            memory_actions.start("alice", "chat")
            memory_actions.start("alice", "chat")
            thread.assert_called_once()
            thread.return_value.start.assert_called_once()
            self.assertTrue(memory_actions.status("alice")["running"])
            self.assertEqual(thread.call_args.kwargs["args"], ("alice", "chat"))

    def test_missing_identity_chat_or_configuration_is_rejected(self):
        for actor, chat in [("", "chat"), ("local-user", "chat"), ("alice", None)]:
            with self.subTest(actor=actor, chat=chat), self.assertRaises(ValueError):
                memory_actions.start(actor, chat)
        with patch("memory_actions.configured", return_value=False), self.assertRaises(ValueError):
            memory_actions.start("alice", "chat")


if __name__ == "__main__":
    unittest.main()
