from datetime import date
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ruamel.yaml import YAML

from scripts import create_genie_agents
from scripts.asana_agents import build_asana_agents
from scripts.generate_asana_data import month_start, qualified_schema


class AsanaSetupTests(unittest.TestCase):
    def test_namespace_is_explicit_and_rejects_sql_injection(self):
        self.assertEqual(qualified_schema("customer_catalog", "asana_demo"),
                         "customer_catalog.asana_demo")
        for catalog, schema in [("", "asana"), ("catalog", "schema; DROP TABLE data"),
                                ("catalog.other", "schema")]:
            with self.assertRaises(ValueError):
                qualified_schema(catalog, schema)

    def test_month_boundaries_cross_years(self):
        self.assertEqual(month_start(date(2025, 11, 1), 12), date(2026, 11, 1))
        self.assertEqual(month_start(date(2025, 11, 1), 8), date(2026, 7, 1))

    def test_agents_are_deterministic_synthetic_and_scoped(self):
        agents = build_asana_agents("demo", "asana", create_genie_agents._col)
        self.assertEqual([list(agent["tables"]) for agent in agents],
                         [["demo.asana.revenue"], ["demo.asana.marketing"]])
        for agent in agents:
            payload = create_genie_agents.build_serialized_space(agent)
            self.assertEqual(payload, create_genie_agents.build_serialized_space(agent))
            self.assertEqual(payload["version"], 2)
            self.assertIn("synthetic", agent["description"])
            self.assertTrue(payload["data_sources"]["tables"][0]["column_configs"])
            self.assertEqual(len(payload["instructions"]["text_instructions"]), 1)
            for _, sql in agent["examples"]:
                self.assertIn("demo.asana.", sql)
                self.assertNotIn("samples.", sql)
        self.assertIn("never across months", str(agents[0]["tables"]))
        self.assertIn("ratios of sums", " ".join(agents[1]["text_instructions"]))

    def test_bundle_switch_is_idempotent_and_preserves_memory(self):
        yaml = YAML()
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "databricks.yml"
            bundle.write_text(Path("databricks.yml").read_text())
            original = yaml.load(bundle.read_text())
            app_key = next(iter(original["resources"]["apps"]))
            memory_resources = [resource for resource in original["resources"]["apps"][app_key]["resources"]
                                if "genie_space" not in resource]
            spaces = [("asana_revenue_genie", "Revenue", "revenue-id"),
                      ("asana_marketing_genie", "Marketing", "marketing-id")]
            with patch.object(create_genie_agents, "DATABRICKS_YML", bundle):
                create_genie_agents.upsert_bundle_resources(spaces, "asana")
                first = bundle.read_text()
                create_genie_agents.upsert_bundle_resources(spaces, "asana")
                self.assertEqual(first, bundle.read_text())
                configured = yaml.load(first)
                app = configured["resources"]["apps"][app_key]
                self.assertEqual([resource for resource in app["resources"] if "genie_space" not in resource],
                                 memory_resources)
                self.assertEqual({resource["name"] for resource in app["resources"] if "genie_space" in resource},
                                 {"asana_revenue_genie", "asana_marketing_genie"})
                environment = {entry["name"]: entry for entry in app["config"]["env"]}
                self.assertNotIn("TPCH_GENIE_SPACE_ID", environment)
                self.assertEqual(environment["ASANA_REVENUE_GENIE_SPACE_ID"]["value_from"], "asana_revenue_genie")
                self.assertEqual(configured["variables"]["genie_dataset"]["default"], "asana")
                create_genie_agents.upsert_bundle_resources(spaces, "asana", claim_views=True)
                claims_config = yaml.load(bundle.read_text())["resources"]["apps"][app_key]
                claim_environment = {entry["name"]: entry for entry in claims_config["config"]["env"]}
                self.assertEqual(claim_environment["GENIE_CUSTOM_CLAIMS"]["value"], "true")
                create_genie_agents.upsert_bundle_resources([
                    ("tpch_orders_genie", "TPC-H", "tpch-id"),
                    ("nyc_taxi_genie", "Taxi", "taxi-id")], "samples")
                reverted = yaml.load(bundle.read_text())["resources"]["apps"][app_key]
                self.assertNotIn("asana_revenue_genie", {resource["name"] for resource in reverted["resources"]})
                sample_environment = {entry["name"]: entry for entry in reverted["config"]["env"]}
                self.assertEqual(sample_environment["GENIE_CUSTOM_CLAIMS"]["value"], "false")

    def test_asana_runtime_does_not_expose_old_sample_agents(self):
        import genie
        try:
            with patch.dict("os.environ", {
                "GENIE_DATASET": "asana", "ASANA_REVENUE_GENIE_SPACE_ID": "revenue-id",
                "ASANA_MARKETING_GENIE_SPACE_ID": "marketing-id", "TPCH_GENIE_SPACE_ID": "old-id",
            }):
                configured = importlib.reload(genie)
                self.assertEqual(configured.SPACES, {"genie_asana_revenue": "revenue-id",
                                                     "genie_asana_marketing": "marketing-id"})
                self.assertIn("synthetic Asana", configured.CHAT_TOPICS)
        finally:
            importlib.reload(genie)


if __name__ == "__main__":
    unittest.main()
