"""Offline contracts for source privacy, consolidation, replay, and Lakeflow wiring."""

import copy
import importlib
import json
import os
import sys
import unittest
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from memory_pipeline.events import process_event, source_events, validate_changes, validate_event
from memory_pipeline.inference import Distiller
from memory_pipeline.store import PostgresRepository, enqueue, ensure_tables


def make_event(actor="alice", text="Use USD for revenue.", source="first", occurred_at=100.0):
    return source_events(actor, {
        "id": "chat-one", "created": occurred_at,
        "events": [{"kind": "user", "text": text, "event_id": source}],
    })[0]


def answer(text="Use USD for revenue.", content="Report revenue in USD.", key="preferences.currency"):
    return json.dumps({"upserts": [{"key": key, "content": content, "evidence": text}],
                       "deletes": []})


class InMemoryRepository:
    def __init__(self):
        self.memories = {}
        self.processed = set()
        self.fail_mark = False

    @contextmanager
    def transaction(self, actor_id):
        snapshot = copy.deepcopy((self.memories, self.processed))
        try:
            yield
        except Exception:
            self.memories, self.processed = snapshot
            raise

    def seen(self, actor_id, event_id):
        return (actor_id, event_id) in self.processed

    def load(self, actor_id):
        return {key: item["content"] for (actor, key), item in self.memories.items()
                if actor == actor_id and not item["deleted"]}

    def apply(self, event, changes):
        for operation, mutations in changes.items():
            for mutation in mutations:
                key = (event["actor_id"], mutation["key"])
                order = (event["occurred_at"], event["event_id"])
                if key not in self.memories or order > self.memories[key]["order"]:
                    self.memories[key] = {"content": mutation.get("content", ""),
                                          "deleted": operation == "deletes", "order": order}

    def mark(self, event):
        if self.fail_mark:
            raise RuntimeError("simulated checkpoint failure")
        self.processed.add((event["actor_id"], event["event_id"]))


class SourceTests(unittest.TestCase):
    def test_only_user_authored_events_are_exported(self):
        chat = {"id": "chat", "created": 100, "events": [
            {"kind": "tool_out", "text": "Use EUR."},
            {"kind": "reply", "text": "Use GBP."},
            {"kind": "user", "text": "Use USD."},
        ]}
        events = source_events("alice", chat)
        self.assertEqual([event["text"] for event in events], ["Use USD."])
        self.assertEqual(events, source_events("alice", chat))

    def test_secrets_and_oversized_messages_are_not_exported(self):
        for text in ["My password is hunter2", "api_key=secret", "x" * 16001, " "]:
            with self.subTest(text=text[:30]):
                chat = {"id": "chat", "created": 100,
                        "events": [{"kind": "user", "text": text}]}
                self.assertEqual(source_events("alice", chat), [])

    def test_branch_copies_reuse_source_ids(self):
        original = {"id": "original", "created": 100, "events": [
            {"kind": "user", "text": "Use USD.", "event_id": "stable", "created": 101}]}
        branch = dict(original, id="branch", branch_of="original")
        self.assertEqual(source_events("alice", original)[0]["event_id"],
                         source_events("alice", branch)[0]["event_id"])

    def test_legacy_branch_history_is_not_learned_again(self):
        branch = {"id": "branch", "created": 100, "branch_of": "parent", "events": [
            {"kind": "user", "text": "Copied legacy history"},
            {"kind": "user", "text": "Use EUR.", "event_id": "new"},
        ]}
        self.assertEqual([event["text"] for event in source_events("alice", branch)],
                         ["Use EUR."])

    def test_actor_is_part_of_stable_identity(self):
        self.assertNotEqual(make_event("alice")["event_id"], make_event("bob")["event_id"])

    def test_legacy_events_preserve_conversation_order(self):
        chat = {"id": "chat", "created": 1793638800.0, "events": [
            {"kind": "user", "text": "Use USD."},
            {"kind": "user", "text": "Use EUR instead."},
        ]}
        first, correction = source_events("alice", chat)
        self.assertLess(first["occurred_at"], correction["occurred_at"])

    def test_invalid_source_events_fail_closed(self):
        for update in [{"actor_id": ""}, {"event_id": "../event"},
                       {"occurred_at": float("inf")}, {"occurred_at": float("nan")},
                       {"occurred_at": True}, {"text": "Bearer token-value"}]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_event(dict(make_event(), **update))


class ConsolidationTests(unittest.TestCase):
    def setUp(self):
        self.repository = InMemoryRepository()
        self.distill = MagicMock(return_value=answer())

    def test_distillation_recall_and_full_refresh_replay(self):
        event = make_event()
        self.assertTrue(process_event(event, self.repository, self.distill))
        self.assertEqual(self.repository.load("alice"),
                         {"preferences.currency": "Report revenue in USD."})
        self.assertFalse(process_event(event, self.repository, self.distill))
        self.distill.assert_called_once()

    def test_existing_memory_is_supplied_for_consolidation(self):
        process_event(make_event(), self.repository, self.distill)
        correction = make_event(text="Use EUR instead.", source="second", occurred_at=200)
        self.distill.return_value = answer("Use EUR instead.", "Report revenue in EUR.")
        process_event(correction, self.repository, self.distill)
        self.assertEqual(self.distill.call_args.args[1],
                         {"preferences.currency": "Report revenue in USD."})
        self.assertEqual(self.repository.load("alice")["preferences.currency"],
                         "Report revenue in EUR.")

    def test_cross_actor_isolation(self):
        process_event(make_event("alice"), self.repository, self.distill)
        self.assertEqual(self.repository.load("bob"), {})
        process_event(make_event("bob"), self.repository, self.distill)
        self.assertEqual(self.distill.call_args.args[1], {})

    def test_out_of_order_sources_do_not_overwrite_newer_facts(self):
        latest = make_event(text="Use EUR instead.", source="latest", occurred_at=200)
        self.distill.return_value = answer("Use EUR instead.", "Report revenue in EUR.")
        process_event(latest, self.repository, self.distill)
        self.distill.return_value = answer()
        process_event(make_event(), self.repository, self.distill)
        self.assertEqual(self.repository.load("alice")["preferences.currency"],
                         "Report revenue in EUR.")

    def test_forgetting_leaves_a_tombstone_against_older_replay(self):
        process_event(make_event(), self.repository, self.distill)
        forgetting = make_event(text="Forget my currency preference.", source="forget", occurred_at=300)
        self.distill.return_value = json.dumps({"upserts": [], "deletes": [{
            "key": "preferences.currency", "evidence": forgetting["text"]}]})
        process_event(forgetting, self.repository, self.distill)
        self.assertEqual(self.repository.load("alice"), {})
        self.distill.return_value = answer()
        process_event(make_event(source="old-replayed", occurred_at=150),
                      self.repository, self.distill)
        self.assertEqual(self.repository.load("alice"), {})

    def test_checkpoint_failure_rolls_back_memories_and_can_retry(self):
        self.repository.fail_mark = True
        with self.assertRaises(RuntimeError):
            process_event(make_event(), self.repository, self.distill)
        self.assertEqual(self.repository.memories, {})
        self.assertEqual(self.repository.processed, set())
        self.repository.fail_mark = False
        self.assertTrue(process_event(make_event(), self.repository, self.distill))

    def test_inference_failure_does_not_advance_checkpoint(self):
        self.distill.side_effect = TimeoutError("model unavailable")
        with self.assertRaises(TimeoutError):
            process_event(make_event(), self.repository, self.distill)
        self.assertEqual(self.repository.processed, set())

    def test_empty_extraction_is_checkpointed(self):
        self.distill.return_value = '{"upserts": [], "deletes": []}'
        self.assertTrue(process_event(make_event(), self.repository, self.distill))
        self.assertFalse(process_event(make_event(), self.repository, self.distill))


class ValidationTests(unittest.TestCase):
    def test_untrusted_model_fields_and_unproven_facts_are_rejected(self):
        invalid = [
            '{"upserts": [], "deletes": [], "actor_id": "bob"}',
            answer(text="The assistant said use EUR."),
            answer(content="api_key=do-not-store"),
            answer(key="INVALID KEY"),
            answer(content=" "),
            '{"upserts": {}, "deletes": []}',
            '{"upserts": [], "deletes": [{"key":"unknown", "evidence":"Use USD"}]}',
            'not JSON',
        ]
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(ValueError):
                validate_changes(response, make_event(), {})

    def test_duplicate_keys_are_rejected(self):
        changes = json.loads(answer())
        changes["upserts"] *= 2
        with self.assertRaises(ValueError):
            validate_changes(json.dumps(changes), make_event(), {})

    def test_memory_growth_is_bounded(self):
        existing = {f"topic.{index}": "Existing fact" for index in range(64)}
        with self.assertRaises(ValueError):
            validate_changes(answer(), make_event(), existing)


class PostgresContractTests(unittest.TestCase):
    def test_identifiers_are_quoted_and_values_parameterized(self):
        conn = MagicMock()
        ensure_tables(conn, 'memory"schema')
        self.assertIn('"memory""schema"."memory_outbox"',
                      conn.execute.call_args_list[1].args[0])
        enqueue(conn, "agent_memory", [make_event()])
        query, params = conn.execute.call_args.args
        self.assertNotIn("alice", query)
        self.assertEqual(params[1], "alice")

    def test_actor_lock_and_replay_lookup_are_scoped(self):
        conn = MagicMock()
        repository = PostgresRepository(conn, "agent_memory")
        with repository.transaction("alice"):
            repository.seen("alice", "event")
        self.assertIn("pg_advisory_xact_lock", conn.execute.call_args_list[0].args[0])
        self.assertEqual(conn.execute.call_args.args[1], ("alice", "event"))

    def test_upsert_guards_source_order_and_retains_provenance(self):
        conn = MagicMock()
        event = make_event()
        changes = validate_changes(answer(), event, {})
        PostgresRepository(conn, "agent_memory").apply(event, changes)
        query, params = conn.execute.call_args.args
        self.assertIn("(EXCLUDED.source_at, EXCLUDED.source_id)", query)
        self.assertEqual(params[0], "alice")
        self.assertEqual(params[3:6], (event["text"], event["event_id"], event["occurred_at"]))


class InferenceTests(unittest.TestCase):
    def test_sdk_timeouts_are_passed_via_config(self):
        with patch("databricks.sdk.WorkspaceClient") as workspace, \
                patch("databricks.sdk.core.Config") as config:
            Distiller("system.ai.gpt-example")
        config.assert_called_once_with(http_timeout_seconds=60, retry_timeout_seconds=120)
        workspace.assert_called_once_with(config=config.return_value)

    def test_empty_native_output_is_rejected(self):
        distill = object.__new__(Distiller)
        distill.client = MagicMock()
        distill.client.api_client.do.return_value = {"output": [{"type": "reasoning"}]}
        distill.model = "system.ai.gpt-example"
        with self.assertRaises(ValueError):
            distill("Use USD.", {})

    def test_gpt_uses_responses_api_with_bounded_output(self):
        distill = object.__new__(Distiller)
        distill.client = MagicMock()
        distill.model = "system.ai.gpt-example"
        distill.client.api_client.do.return_value = {"output": [{
            "type": "message", "content": [{"type": "output_text", "text": answer()}]}]}
        self.assertEqual(distill("Use USD.", {}), answer())
        self.assertEqual(distill.client.api_client.do.call_args.args,
                         ("POST", "/ai-gateway/openai/v1/responses"))
        request = distill.client.api_client.do.call_args.kwargs["body"]
        self.assertEqual(request["max_output_tokens"], 4096)
        self.assertEqual(request["input"][0]["role"], "user")

    def test_existing_and_new_content_are_untrusted_user_data(self):
        distill = object.__new__(Distiller)
        distill.client = MagicMock()
        distill.model = "chosen.model"
        distill.client.api_client.do.return_value = {"choices": [{
            "message": {"content": answer()}}]}
        self.assertEqual(distill("Use USD.", {"topic": "fact"}), answer())
        self.assertEqual(distill.client.api_client.do.call_args.args,
                         ("POST", "/ai-gateway/mlflow/v1/chat/completions"))
        request = distill.client.api_client.do.call_args.kwargs["body"]
        self.assertEqual(request["model"], "chosen.model")
        self.assertEqual(request["messages"][1]["role"], "user")
        self.assertEqual(json.loads(request["messages"][1]["content"])["existing_memories"],
                         {"topic": "fact"})


class GraphTests(unittest.TestCase):
    def test_graph_registers_real_tables_and_sink_without_executing_inference(self):
        spark = MagicMock()
        spark.conf.get.side_effect = lambda name, default=None: {
            "memory.source_path": "/Volumes/catalog/schema/volume/events",
            "memory.lakebase_endpoint": "projects/project/branches/branch/endpoints/primary",
            "memory.model": "chosen.model",
        }.get(name, default)
        registrations = []

        def register(kind):
            def decorate(**kwargs):
                def capture(function):
                    registrations.append((kind, kwargs, function))
                    return function
                return capture
            return decorate

        pipelines = SimpleNamespace(table=register("table"),
                                    foreach_batch_sink=register("sink"),
                                    append_flow=register("flow"),
                                    expect_or_fail=lambda *args: lambda function: function)
        pyspark = ModuleType("pyspark")
        pyspark.pipelines = pipelines
        sql_module = ModuleType("pyspark.sql")
        sql_module.SparkSession = SimpleNamespace(getActiveSession=lambda: spark)
        functions = ModuleType("pyspark.sql.functions")
        functions.col = lambda name: name
        with patch.dict(sys.modules, {"pyspark": pyspark, "pyspark.sql": sql_module,
                                      "pyspark.sql.functions": functions}), \
                patch("memory_pipeline.inference.Distiller") as distiller:
            sys.modules.pop("memory_pipeline.definitions", None)
            try:
                graph = importlib.import_module("memory_pipeline.definitions")
                self.assertEqual([kind for kind, _, _ in registrations],
                                 ["table", "table", "sink", "flow"])
                distiller.assert_not_called()
                graph.bronze_memory_events()
                spark.readStream.format.assert_called_once_with("cloudFiles")
                graph.publish_long_term_memories()
                spark.readStream.table.assert_called_with("silver_memory_inputs")
                event = make_event()
                batch = MagicMock()
                batch.orderBy.return_value.limit.return_value.collect.side_effect = lambda: [
                    SimpleNamespace(asDict=lambda: dict(event))]
                repository = InMemoryRepository()
                extract = MagicMock(return_value=answer())
                distiller.return_value = extract
                with patch("databricks.sdk.WorkspaceClient"), \
                        patch("memory_pipeline.connection.connect"), \
                        patch("memory_pipeline.store.ensure_tables"), \
                        patch("memory_pipeline.store.PostgresRepository", return_value=repository):
                    graph.consolidate_long_term_memory(batch, 0)
                    graph.consolidate_long_term_memory(batch, 99)
                    batch.orderBy.return_value.limit.assert_called_with(101)
                    batch.orderBy.return_value.limit.return_value.collect.side_effect = None
                    batch.orderBy.return_value.limit.return_value.collect.return_value = [event] * 101
                    with self.assertRaises(ValueError):
                        graph.consolidate_long_term_memory(batch, 100)
                extract.assert_called_once()
                self.assertEqual(repository.load("alice"),
                                 {"preferences.currency": "Report revenue in USD."})
            finally:
                sys.modules.pop("memory_pipeline.definitions", None)


class PipelineConnectionTests(unittest.TestCase):
    def test_transaction_commits_only_after_success(self):
        from memory_pipeline.connection import PipelineConnection

        raw = MagicMock()
        connection = PipelineConnection(raw)
        with connection.transaction():
            connection.execute("SELECT %s", ("alice",))
        calls = raw.cursor.return_value.execute.call_args_list
        self.assertEqual([entry.args[0] for entry in calls], ["BEGIN", "SELECT %s", "COMMIT"])
        self.assertEqual(calls[1].args[1], ("alice",))

    def test_transaction_rolls_back_failures(self):
        from memory_pipeline.connection import PipelineConnection

        raw = MagicMock()
        with self.assertRaises(RuntimeError), PipelineConnection(raw).transaction():
            raise RuntimeError("inference failed")
        self.assertEqual([entry.args[0] for entry in raw.cursor.return_value.execute.call_args_list],
                         ["BEGIN", "ROLLBACK"])


if __name__ == "__main__":
    unittest.main()
