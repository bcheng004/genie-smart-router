"""Contracts for the durable ingestion bridge and application recall integration."""

import io
import json
import os
import unittest
from unittest.mock import MagicMock, patch

from databricks.sdk.errors import AlreadyExists, ResourceAlreadyExists

import long_term_memory
import memory
from tests.test_memory_pipeline import make_event


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "LONG_TERM_MEMORY_VOLUME": "/Volumes/catalog/schema/volume",
            "LAKEBASE_ENDPOINT": "projects/project/branches/branch/endpoints/primary",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.connection = MagicMock()
        self.client = MagicMock()
        self.runner = patch.object(memory, "_run", side_effect=lambda function: function(self.connection))
        self.runner.start()
        self.addCleanup(self.runner.stop)
        self.workspace = patch.object(memory, "_workspace_client", return_value=self.client)
        self.workspace.start()
        self.addCleanup(self.workspace.stop)
        self.durable = patch.object(memory, "use_durable", return_value=True)
        self.durable.start()
        self.addCleanup(self.durable.stop)

    def test_immutable_publication_is_marked_only_after_upload(self):
        event = make_event()
        self.connection.execute.return_value.fetchall.return_value = [(event["event_id"], event)]
        self.assertEqual(long_term_memory.publish_pending(), 1)
        args = self.client.files.upload.call_args
        self.assertTrue(args.args[0].endswith(event["event_id"] + ".json"))
        self.assertIsInstance(args.args[1], io.BytesIO)
        self.assertEqual(args.kwargs, {"overwrite": False})
        self.assertIn("SET published = true", self.connection.execute.call_args.args[0].as_string())

    def test_crash_after_upload_is_safe_to_retry(self):
        event = make_event()
        self.connection.execute.return_value.fetchall.return_value = [(event["event_id"], event)]
        for error in (AlreadyExists, ResourceAlreadyExists):
            with self.subTest(error=error):
                self.client.files.upload.side_effect = error("file previously uploaded")
                self.assertEqual(long_term_memory.publish_pending(), 1)
                self.assertIn("SET published = true", self.connection.execute.call_args.args[0].as_string())

    def test_upload_failure_does_not_mark_publication(self):
        event = make_event()
        self.connection.execute.return_value.fetchall.return_value = [(event["event_id"], event)]
        self.client.files.upload.side_effect = TimeoutError("volume unavailable")
        with self.assertRaises(TimeoutError):
            long_term_memory.publish_pending()
        queries = [call.args[0].as_string() if hasattr(call.args[0], "as_string") else call.args[0]
                   for call in self.connection.execute.call_args_list]
        self.assertFalse(any("SET published = true" in query for query in queries))

    def test_manual_publication_is_scoped_to_actor_and_conversation(self):
        self.connection.execute.return_value.fetchall.return_value = []
        long_term_memory.publish_pending(actor_id="alice", chat_id="chat")
        query, params = self.connection.execute.call_args.args
        self.assertIn("actor_id = %s", query.as_string())
        self.assertIn("payload ->> 'chat_id' = %s", query.as_string())
        self.assertEqual(params, ("alice", "chat", 100))

    def test_manual_queue_checks_stored_conversation_ownership(self):
        self.connection.execute.return_value.fetchone.return_value = None
        with self.assertRaises(ValueError):
            long_term_memory.queue_chat("alice", "bobs-chat")
        query, params = self.connection.execute.call_args.args
        self.assertIn("chat_id = %s AND app_user = %s", query.as_string())
        self.assertEqual(params, ("bobs-chat", "alice"))

    def test_manual_queue_exports_only_eligible_messages_from_selected_chat(self):
        self.connection.execute.return_value.fetchone.return_value = (100, None, [
            {"kind": "user", "text": "Use USD.", "event_id": "source"},
            {"kind": "reply", "text": "Assistant output"},
        ])
        self.assertEqual(long_term_memory.queue_chat("alice", "chat"), 1)
        query, params = self.connection.execute.call_args.args
        self.assertIn("memory_outbox", query)
        self.assertEqual(params[1], "alice")
        self.assertEqual(json.loads(params[2])["chat_id"], "chat")

    def test_manual_mode_never_automatically_exports_chat_settlement(self):
        with patch.dict(os.environ, {"LONG_TERM_MEMORY_AUTO_EXPORT": "false"}):
            memory.save_chat("alice", {"id": "chat", "title": "Revenue", "created": 100,
                                      "updated": 101, "events": []})
            queries = [call.args[0] for call in self.connection.execute.call_args_list]
            self.assertEqual(len(queries), 1)
            self.assertIn(".chats", queries[0])
            with patch("long_term_memory.threading.Thread") as thread:
                long_term_memory.start_publisher()
                thread.assert_not_called()

    def test_recall_is_parameterized_and_scoped_to_authenticated_actor(self):
        self.connection.execute.return_value.fetchall.return_value = [("currency", "Use USD.")]
        self.assertEqual(long_term_memory.recall("alice"), [{"key": "currency", "content": "Use USD."}])
        query, params = self.connection.execute.call_args.args
        self.assertEqual(params, ("alice", 8))
        self.assertIn("actor_id = %s AND NOT deleted", query.as_string())

    def test_recall_filters_suspected_secrets(self):
        self.connection.execute.return_value.fetchall.return_value = [("bad", "api_key=secret")]
        self.assertEqual(long_term_memory.recall("alice"), [])

    def test_optional_feature_is_disabled_without_volume(self):
        with patch.dict(os.environ, {"LONG_TERM_MEMORY_VOLUME": ""}):
            self.assertEqual(long_term_memory.publish_pending(), 0)
            self.assertEqual(long_term_memory.recall("alice"), [])
            self.client.files.upload.assert_not_called()

    def test_invalid_volume_paths_are_rejected(self):
        for path in ["/tmp/events", "/Volumes/catalog/schema/volume/events",
                     "/Volumes//schema/volume", "/Volumes/catalog/../volume"]:
            with self.subTest(path=path), patch.dict(os.environ, {"LONG_TERM_MEMORY_VOLUME": path}):
                with self.assertRaises(ValueError):
                    long_term_memory.source_path()

    def test_chat_and_export_events_share_a_transaction(self):
        record = {"id": "chat", "title": "Revenue", "created": 100, "updated": 101,
                  "events": [{"kind": "user", "text": "Use USD.", "event_id": "event"}]}
        memory.save_chat("alice", record)
        self.connection.transaction.assert_called_once()
        queries = [call.args[0].as_string() if hasattr(call.args[0], "as_string") else call.args[0]
                   for call in self.connection.execute.call_args_list]
        self.assertIn('INSERT INTO "agent_memory".chats', queries[0])
        self.assertIn('INSERT INTO "agent_memory"."memory_outbox"', queries[-1])


if __name__ == "__main__":
    unittest.main()
