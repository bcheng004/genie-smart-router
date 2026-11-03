"""Exercise the agent's real run function without workspace or model calls."""

import importlib
import json
import os
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


class RecallIntegrationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {
            "DATABRICKS_HOST": "https://offline.invalid", "DATABRICKS_TOKEN": "offline-test-token",
            "DATABRICKS_AUTH_TYPE": "pat",
        }), patch("agents.set_default_openai_client"), patch("agents.set_default_openai_api"), \
                patch("databricks_openai.utils.clients.WorkspaceClient") as workspace:
            workspace.return_value.config.host = "https://offline.invalid"
            cls.agent = importlib.import_module("agent")

    async def asyncSetUp(self):
        @asynccontextmanager
        async def no_servers(*args, **kwargs):
            yield []

        self.servers = patch.object(self.agent.genie, "mcp_servers", no_servers)
        self.servers.start()
        self.addCleanup(self.servers.stop)
        self.model = patch.object(self.agent, "_model_for", return_value="offline-model")
        self.model.start()
        self.addCleanup(self.model.stop)
        self.runner = patch.object(self.agent.Runner, "run", new_callable=AsyncMock,
                                   return_value=SimpleNamespace(final_output="Answer"))
        self.run_mock = self.runner.start()
        self.addCleanup(self.runner.stop)
        self.recall = patch("long_term_memory.recall", return_value=[{
            "key": "preferences.currency", "content": "Use USD for revenue.",
        }])
        self.recall_mock = self.recall.start()
        self.addCleanup(self.recall.stop)

    async def test_new_chat_recalls_only_authenticated_actor_as_user_context(self):
        reply, _, _ = await self.agent.run("Show revenue", self.agent.MemorySession(),
                                           model="offline-model", actor_id="alice")
        self.assertEqual(reply, "Answer")
        self.recall_mock.assert_called_once_with("alice")
        inputs = self.run_mock.call_args.kwargs["input"]
        self.assertEqual([item["role"] for item in inputs], ["user", "user"])
        self.assertEqual(inputs[1]["content"], "Show revenue")
        self.assertEqual(json.loads(inputs[0]["content"].split("\n", 1)[1])[0]["key"],
                         "preferences.currency")

    async def test_existing_chat_does_not_accumulate_recalled_context(self):
        session = self.agent.MemorySession()
        await session.add_items([{"role": "user", "content": "Earlier question"}])
        await self.agent.run("Show revenue", session, model="offline-model", actor_id="alice")
        self.recall_mock.assert_not_called()
        self.assertEqual(self.run_mock.call_args.kwargs["input"], "Show revenue")

    async def test_no_actor_means_no_cross_user_recall(self):
        await self.agent.run("Show revenue", self.agent.MemorySession(), model="offline-model")
        self.recall_mock.assert_not_called()

    async def test_empty_recall_preserves_original_input(self):
        self.recall_mock.return_value = []
        await self.agent.run("Show revenue", self.agent.MemorySession(),
                             model="offline-model", actor_id="alice")
        self.assertEqual(self.run_mock.call_args.kwargs["input"], "Show revenue")


if __name__ == "__main__":
    unittest.main()
