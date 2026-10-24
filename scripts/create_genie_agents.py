#!/usr/bin/env python3
"""Create (or update) two Genie agents over Databricks sample datasets.

  * "TPC-H Order Analytics" — samples.tpch customer / orders / lineitem / nation / region
  * "NYC Taxi Trips"        — samples.nyctaxi.trips

Adapted from scripts/create_genie_space.py in
https://github.com/bcheng004/databricks-agent-on-apps-genie-mcp : agents are matched by title,
so reruns update the existing Genie agent instead of creating duplicates, and each agent is
upserted as a `genie_space` app resource (CAN_RUN) in databricks.yml.

Usage:
    uv run create-genie-agents --profile <profile> --dry-run    # print payloads
    uv run create-genie-agents --profile <profile> --validate   # check the SQL
    uv run create-genie-agents --profile <profile>              # create/update
"""
import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import State, StatementState

DATABRICKS_YML = Path(__file__).resolve().parent.parent / "databricks.yml"
RT_WAREHOUSE_NAME = "rt-warehouse"


def ensure_realtime_warehouse(w: WorkspaceClient, name: str = RT_WAREHOUSE_NAME) -> str:
    """Find-or-create the workspace's REALTIME SQL warehouse (low-latency serving) and return
    its id. Reruns reuse the warehouse found by name.

    Created through the raw API: this SDK's WarehouseType enum predates REALTIME (a
    SimpleNamespace(value=...) would pass it through, but the create waiter treats the
    warehouse's rest state STOPPED as a failure, so create_and_wait can't be used)."""
    for wh in w.warehouses.list():
        if wh.name == name:
            return wh.id
    res = w.api_client.do("POST", "/api/2.0/sql/warehouses", body={
        "name": name, "cluster_size": "Small", "max_num_clusters": 1,
        "enable_serverless_compute": True,
        "warehouse_type": "REALTIME",  # newer than this SDK's WarehouseType enum
    })
    wh_id = res["id"]
    for _ in range(30):  # rest state is STOPPED (serverless): start happens on first query
        if w.warehouses.get(wh_id).state in (State.RUNNING, State.STOPPED):
            break
        time.sleep(5)
    print(f"  -> realtime warehouse {name!r}: {wh_id}")
    return wh_id


def _id(*parts: str) -> str:
    """Stable 32-hex id, so reruns produce the same serialized_space."""
    return uuid.uuid5(uuid.NAMESPACE_URL, "genie-agent/" + "/".join(parts)).hex


def _col(name, description=None, synonyms=None, match=False, exclude=False):
    c = {"column_name": name}
    if description:
        c["description"] = [description]
    if synonyms:
        c["synonyms"] = synonyms
    if match:  # only for low-cardinality categorical strings users name directly
        c["enable_format_assistance"] = True
        c["enable_entity_matching"] = True
    if exclude:
        c["exclude"] = True
    return c


# Each agent: Genie surfaces plus `validate_from` per snippet (the FROM clause used to check it).
TPCH = "samples.tpch"
AGENTS = [
    {
        "key": "tpch_orders_genie",
        "title": "TPC-H Order Analytics (Router)",
        "description": (
            "Answers questions about B2B orders, revenue, customers, shipping and returns in the "
            "TPC-H sample dataset (samples.tpch): customers by market segment, nation and region; "
            "orders by status and priority; line items by ship mode and return flag. "
            "Orders span 1992-01-01 to 1998-08-02."
        ),
        "tables": {
            f"{TPCH}.customer": [
                _col("c_acctbal", "Customer account balance.", ["account balance", "balance"]),
                _col("c_address", exclude=True),
                _col("c_comment", exclude=True),
                _col("c_custkey", "Customer key; joins to orders.o_custkey."),
                _col("c_mktsegment", "Customer market segment: AUTOMOBILE, BUILDING, FURNITURE, "
                     "HOUSEHOLD or MACHINERY.", ["market segment", "segment", "industry"],
                     match=True),
                _col("c_name", "Customer name, e.g. Customer#000412445.", ["customer"]),
                _col("c_nationkey", "Nation key; joins to nation.n_nationkey."),
                _col("c_phone", exclude=True),
            ],
            f"{TPCH}.lineitem": [
                _col("l_comment", exclude=True),
                _col("l_discount", "Discount as a fraction (0.00-0.10).", ["discount"]),
                _col("l_extendedprice", "Line gross price (quantity x part price), before "
                     "discount.", ["gross price", "extended price"]),
                _col("l_orderkey", "Order key; joins to orders.o_orderkey."),
                _col("l_quantity", "Units ordered on the line.", ["quantity", "units"]),
                _col("l_returnflag", "Return flag: R = returned, A = accepted, N = not yet "
                     "returnable.", ["return flag", "returned"], match=True),
                _col("l_shipdate", "Date the line item shipped.", ["ship date"]),
                _col("l_shipinstruct", "Shipping instruction.", match=True),
                _col("l_shipmode", "Ship mode: AIR, FOB, MAIL, RAIL, REG AIR, SHIP or TRUCK.",
                     ["ship mode", "shipping method", "carrier"], match=True),
            ],
            f"{TPCH}.nation": [
                _col("n_comment", exclude=True),
                _col("n_name", "Nation (country) name, e.g. CANADA.", ["country", "nation"],
                     match=True),
                _col("n_regionkey", "Region key; joins to region.r_regionkey."),
            ],
            f"{TPCH}.orders": [
                _col("o_clerk", exclude=True),
                _col("o_comment", exclude=True),
                _col("o_custkey", "Customer key; joins to customer.c_custkey."),
                _col("o_orderdate", "Date the order was placed.", ["order date"]),
                _col("o_orderkey", "Order key; joins to lineitem.l_orderkey."),
                _col("o_orderpriority", "Order priority: 1-URGENT, 2-HIGH, 3-MEDIUM, "
                     "4-NOT SPECIFIED, 5-LOW.", ["priority"], match=True),
                _col("o_orderstatus", "Order status: F = fulfilled, O = open, P = partially "
                     "fulfilled.", ["status"], match=True),
                _col("o_totalprice", "Total order price.", ["order value", "order total"]),
            ],
            f"{TPCH}.region": [
                _col("r_comment", exclude=True),
                _col("r_name", "Region name: AFRICA, AMERICA, ASIA, EUROPE or MIDDLE EAST.",
                     ["region"], match=True),
            ],
        },
        "joins": [
            ("customer", "orders", "`customer`.`c_custkey` = `orders`.`o_custkey`", "ONE_TO_MANY"),
            ("orders", "lineitem", "`orders`.`o_orderkey` = `lineitem`.`l_orderkey`",
             "ONE_TO_MANY"),
            ("customer", "nation", "`customer`.`c_nationkey` = `nation`.`n_nationkey`",
             "MANY_TO_ONE"),
            ("nation", "region", "`nation`.`n_regionkey` = `region`.`r_regionkey`",
             "MANY_TO_ONE"),
        ],
        "measures": [
            {"alias": "revenue", "display_name": "Revenue",
             "sql": "SUM(lineitem.l_extendedprice * (1 - lineitem.l_discount))",
             "synonyms": ["net revenue", "sales", "discounted revenue"],
             "validate_from": f"{TPCH}.lineitem AS lineitem"},
            {"alias": "order_count", "display_name": "Order count",
             "sql": "COUNT(DISTINCT orders.o_orderkey)", "synonyms": ["number of orders"],
             "validate_from": f"{TPCH}.orders AS orders"},
        ],
        "filters": [
            {"display_name": "Returned line items", "sql": "lineitem.l_returnflag = 'R'",
             "synonyms": ["returns", "returned"], "validate_from": f"{TPCH}.lineitem AS lineitem"},
        ],
        "expressions": [
            {"alias": "order_year", "display_name": "Order year",
             "sql": "YEAR(orders.o_orderdate)", "validate_from": f"{TPCH}.orders AS orders"},
        ],
        "examples": [
            ("What was revenue by region in 1996?",
             f"""SELECT region.r_name AS region,
       SUM(lineitem.l_extendedprice * (1 - lineitem.l_discount)) AS revenue
FROM {TPCH}.lineitem AS lineitem
JOIN {TPCH}.orders AS orders ON orders.o_orderkey = lineitem.l_orderkey
JOIN {TPCH}.customer AS customer ON customer.c_custkey = orders.o_custkey
JOIN {TPCH}.nation AS nation ON nation.n_nationkey = customer.c_nationkey
JOIN {TPCH}.region AS region ON region.r_regionkey = nation.n_regionkey
WHERE YEAR(orders.o_orderdate) = 1996
GROUP BY region.r_name
ORDER BY revenue DESC"""),
            ("Who were the top 10 customers by total order value in 1997?",
             f"""SELECT customer.c_name, customer.c_mktsegment,
       SUM(orders.o_totalprice) AS total_order_value
FROM {TPCH}.orders AS orders
JOIN {TPCH}.customer AS customer ON customer.c_custkey = orders.o_custkey
WHERE YEAR(orders.o_orderdate) = 1997
GROUP BY customer.c_name, customer.c_mktsegment
ORDER BY total_order_value DESC
LIMIT 10"""),
            ("What is the return rate by ship mode?",
             f"""SELECT lineitem.l_shipmode,
       ROUND(100.0 * SUM(CASE WHEN lineitem.l_returnflag = 'R' THEN 1 ELSE 0 END) / COUNT(*), 2)
         AS return_rate_pct
FROM {TPCH}.lineitem AS lineitem
GROUP BY lineitem.l_shipmode
ORDER BY return_rate_pct DESC"""),
            ("How many orders were placed each month in 1995?",
             f"""SELECT DATE_TRUNC('MONTH', orders.o_orderdate) AS order_month,
       COUNT(DISTINCT orders.o_orderkey) AS order_count
FROM {TPCH}.orders AS orders
WHERE YEAR(orders.o_orderdate) = 1995
GROUP BY DATE_TRUNC('MONTH', orders.o_orderdate)
ORDER BY order_month"""),
        ],
        "sample_questions": [
            "What was revenue by region in 1996?",
            "Which market segment generates the most revenue?",
            "Who were the top 10 customers by order value in 1997?",
            "Which ship mode has the highest return rate?",
            "How many urgent orders are still open?",
        ],
        "text_instructions": [
            "## DISAMBIGUATION",
            "- Orders span 1992-01-01 to 1998-08-02. If the user says 'this year', 'recent' or "
            "'latest' without a year, use 1998 and say it is a partial year.",
            "",
            "## Instructions you must follow when providing summaries",
            "- Always state the date range used.",
        ],
    },
    {
        "key": "nyc_taxi_genie",
        "title": "NYC Taxi Trips (Router)",
        "description": (
            "Answers questions about NYC yellow taxi trips in the samples.nyctaxi.trips sample "
            "dataset (21,932 trips, January-February 2016): trip counts, fares, distances, "
            "durations, and pickup/dropoff ZIP codes."
        ),
        "tables": {
            "samples.nyctaxi.trips": [
                _col("dropoff_zip", "ZIP code where the trip ended.",
                     ["dropoff zip", "destination", "drop-off location"]),
                _col("fare_amount", "Metered fare in USD (excludes tips and tolls).",
                     ["fare", "price", "cost"]),
                _col("pickup_zip", "ZIP code where the trip started.",
                     ["pickup zip", "origin", "pickup location"]),
                _col("tpep_dropoff_datetime", "When the meter was disengaged.",
                     ["dropoff time", "end time"]),
                _col("tpep_pickup_datetime", "When the meter was engaged.",
                     ["pickup time", "start time", "trip date"]),
                _col("trip_distance", "Trip distance in miles.", ["distance", "miles"]),
            ],
        },
        "joins": [],
        "measures": [
            {"alias": "trip_count", "display_name": "Trip count", "sql": "COUNT(*)",
             "synonyms": ["number of trips", "rides"],
             "validate_from": "samples.nyctaxi.trips AS trips"},
            {"alias": "avg_fare", "display_name": "Average fare", "sql": "AVG(trips.fare_amount)",
             "synonyms": ["mean fare", "typical fare"],
             "validate_from": "samples.nyctaxi.trips AS trips"},
        ],
        "filters": [
            {"display_name": "Valid trips",
             "sql": "trips.fare_amount > 0 AND trips.trip_distance > 0",
             "synonyms": ["valid", "clean trips"],
             "validate_from": "samples.nyctaxi.trips AS trips"},
        ],
        "expressions": [
            {"alias": "trip_duration_minutes", "display_name": "Trip duration (minutes)",
             "sql": "(UNIX_TIMESTAMP(trips.tpep_dropoff_datetime) - "
                    "UNIX_TIMESTAMP(trips.tpep_pickup_datetime)) / 60.0",
             "validate_from": "samples.nyctaxi.trips AS trips"},
            {"alias": "pickup_hour", "display_name": "Pickup hour",
             "sql": "HOUR(trips.tpep_pickup_datetime)",
             "validate_from": "samples.nyctaxi.trips AS trips"},
        ],
        "examples": [
            ("What are the 10 busiest pickup ZIP codes?",
             """SELECT trips.pickup_zip, COUNT(*) AS trip_count
FROM samples.nyctaxi.trips AS trips
GROUP BY trips.pickup_zip
ORDER BY trip_count DESC
LIMIT 10"""),
            ("What is the average fare by hour of day?",
             """SELECT HOUR(trips.tpep_pickup_datetime) AS pickup_hour,
       ROUND(AVG(trips.fare_amount), 2) AS avg_fare
FROM samples.nyctaxi.trips AS trips
WHERE trips.fare_amount > 0 AND trips.trip_distance > 0
GROUP BY HOUR(trips.tpep_pickup_datetime)
ORDER BY pickup_hour"""),
            ("How many trips were there per day?",
             """SELECT DATE(trips.tpep_pickup_datetime) AS trip_date, COUNT(*) AS trip_count
FROM samples.nyctaxi.trips AS trips
GROUP BY DATE(trips.tpep_pickup_datetime)
ORDER BY trip_date"""),
            ("What is the average fare per mile?",
             """SELECT ROUND(SUM(trips.fare_amount) / SUM(trips.trip_distance), 2) AS fare_per_mile
FROM samples.nyctaxi.trips AS trips
WHERE trips.fare_amount > 0 AND trips.trip_distance > 0"""),
        ],
        "sample_questions": [
            "What are the busiest pickup ZIP codes?",
            "What is the average fare by hour of day?",
            "How many trips happened per day in February 2016?",
            "What is the average trip duration in minutes?",
            "Which dropoff ZIP has the highest average fare?",
        ],
        "text_instructions": [
            "## DATA QUALITY NOTES",
            "- 10 trips have fare_amount <= 0 and 76 have trip_distance = 0; exclude them "
            "(the 'Valid trips' filter) when computing averages, rates or per-mile figures.",
            "- Data covers 2016-01-01 to 2016-02-29 only.",
        ],
    },
]


def build_serialized_space(agent: dict) -> dict:
    key = agent["key"]
    tables = [{"identifier": ident,
               "column_configs": sorted(cols, key=lambda c: c["column_name"])}
              for ident, cols in sorted(agent["tables"].items())]
    by_alias = {ident.rsplit(".", 1)[-1]: ident for ident in agent["tables"]}

    def snippets(kind):
        out = []
        for s in agent[kind]:
            item = {"id": _id(key, kind, s["display_name"]), "display_name": s["display_name"],
                    "sql": [s["sql"]]}
            if "alias" in s:
                item["alias"] = s["alias"]
            if s.get("synonyms"):
                item["synonyms"] = s["synonyms"]
            out.append(item)
        return sorted(out, key=lambda x: x["id"])

    instructions = {
        "example_question_sqls": sorted(
            [{"id": _id(key, "example", q), "question": [q], "sql": [sql]}
             for q, sql in agent["examples"]], key=lambda x: x["id"]),
        "text_instructions": [{"id": _id(key, "text"), "content": agent["text_instructions"]}],
        "sql_snippets": {k: snippets(k) for k in ("measures", "filters", "expressions")},
    }
    if agent["joins"]:
        instructions["join_specs"] = sorted(
            [{"id": _id(key, "join", left, right),
              "left": {"identifier": by_alias[left], "alias": left},
              "right": {"identifier": by_alias[right], "alias": right},
              "sql": [cond, f"--rt=FROM_RELATIONSHIP_TYPE_{rel}--"]}
             for left, right, cond, rel in agent["joins"]], key=lambda x: x["id"])

    return {
        "version": 2,
        "config": {"sample_questions": sorted(
            [{"id": _id(key, "sample", q), "question": [q]} for q in agent["sample_questions"]],
            key=lambda x: x["id"])},
        "data_sources": {"tables": tables},
        "instructions": instructions,
    }


def validate(w: WorkspaceClient, warehouse_id: str) -> bool:
    """Read-only check: run every example SQL (LIMIT 5) and every snippet against its table."""
    checks = []
    for agent in AGENTS:
        for q, sql in agent["examples"]:
            checks.append((f"{agent['title']} · example · {q}",
                           f"SELECT * FROM ({sql}) LIMIT 5"))
        for s in agent["measures"] + agent["expressions"]:
            checks.append((f"{agent['title']} · snippet · {s['display_name']}",
                           f"SELECT {s['sql']} AS v FROM {s['validate_from']} LIMIT 5"))
        for s in agent["filters"]:
            checks.append((f"{agent['title']} · filter · {s['display_name']}",
                           f"SELECT COUNT(*) AS v FROM {s['validate_from']} WHERE {s['sql']}"))
    ok = True
    for label, sql in checks:
        r = w.statement_execution.execute_statement(statement=sql, warehouse_id=warehouse_id,
                                                    wait_timeout="50s")
        state = r.status.state if r.status else None
        # realtime warehouses answer in ARROW_STREAM, which this SDK doesn't inline into
        # data_array; row_count still comes from the server, so fall back to it
        rows = len(r.result.data_array or []) if r.result else 0
        if r.result is not None and not rows:
            rows = r.result.row_count or 0
        if state == StatementState.SUCCEEDED and rows:
            print(f"  ok   {label} ({rows} rows)")
        else:
            ok = False
            err = r.status.error.message if r.status and r.status.error else f"{rows} rows"
            print(f"  FAIL {label}: {state} {err}")
    return ok


def resolve_warehouse(w: WorkspaceClient, warehouse_id: str | None) -> str | None:
    """--warehouse-id, else the workspace's realtime warehouse (created if missing): used for
    the example-SQL validation, which any running warehouse serves."""
    if warehouse_id:
        return warehouse_id
    return ensure_realtime_warehouse(w)


def resolve_genie_warehouse(w: WorkspaceClient, warehouse_id: str | None) -> str | None:
    """--warehouse-id, else the workspace's first PRO or SERVERLESS warehouse: the only types
    Genie runs on (it rejects REALTIME with "Please switch to a Pro or Serverless warehouse",
    which the Genie MCP server masks as a generic tool error)."""
    if warehouse_id:
        return warehouse_id
    for wh in w.warehouses.list():
        if wh.warehouse_type in ("PRO", "SERVERLESS"):
            return wh.id
    return None


def find_space(w: WorkspaceClient, title: str):
    page_token = None
    while True:
        resp = w.genie.list_spaces(page_token=page_token)
        for space in resp.spaces or []:
            if space.title == title:
                return space
        page_token = resp.next_page_token
        if not page_token:
            return None


def upsert_bundle_resources(spaces: list[tuple[str, str, str]]) -> None:
    """Add/replace `genie_space` app resources (CAN_RUN) in databricks.yml, keyed by resource
    name; other resources are left untouched."""
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    cfg = yaml.load(DATABRICKS_YML.read_text())
    apps = cfg.get("resources", {}).get("apps", {})
    if not apps:
        print(f"  (no resources.apps in {DATABRICKS_YML}; skipping)")
        return
    app_key = next(iter(apps))
    resources = apps[app_key].setdefault("resources", [])
    for name, title, space_id in spaces:
        entry = CommentedMap()
        entry["name"] = name
        entry["genie_space"] = CommentedMap(
            [("name", title), ("space_id", space_id), ("permission", "CAN_RUN")])
        idx = next((i for i, r in enumerate(resources) if r.get("name") == name), None)
        if idx is None:
            resources.append(entry)
        else:
            resources[idx] = entry
        print(f"  -> databricks.yml: app '{app_key}' resource {name!r} -> {space_id}")
    with DATABRICKS_YML.open("w") as f:
        yaml.dump(cfg, f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default=os.getenv("DATABRICKS_CONFIG_PROFILE"),
                        help="Databricks CLI profile (default: $DATABRICKS_CONFIG_PROFILE)")
    parser.add_argument("--warehouse-id",
                        help="SQL warehouse (default: the realtime warehouse for example-SQL "
                             "validation; the Genie agents get the workspace's first PRO or "
                             "SERVERLESS warehouse — Genie can't run on REALTIME)")
    parser.add_argument("--parent-path", help="Workspace folder for the Genie agents")
    parser.add_argument("--dry-run", action="store_true", help="Print payloads; change nothing")
    parser.add_argument("--validate", action="store_true",
                        help="Run the example SQL and snippets read-only; change nothing")
    parser.add_argument("--no-bundle", action="store_true",
                        help="Don't add genie_space resources to databricks.yml")
    args = parser.parse_args()

    if args.dry_run:
        for agent in AGENTS:
            print(f"# {agent['title']}\n# {agent['description']}")
            print(json.dumps(build_serialized_space(agent), indent=2))
        return
    if not args.profile:
        sys.exit("Error: pass --profile or set DATABRICKS_CONFIG_PROFILE")

    w = WorkspaceClient(profile=args.profile)
    warehouse_id = resolve_warehouse(w, args.warehouse_id)
    if not warehouse_id:
        sys.exit("Error: could not resolve a SQL warehouse; pass --warehouse-id")
    print(f"Workspace: {w.config.host}  warehouse: {warehouse_id}")

    if args.validate:
        sys.exit(0 if validate(w, warehouse_id) else 1)

    genie_warehouse_id = resolve_genie_warehouse(w, args.warehouse_id)
    if not genie_warehouse_id:
        sys.exit("Error: no PRO or SERVERLESS warehouse for the Genie agents (Genie can't run "
                 "on REALTIME warehouses); pass --warehouse-id")
    print(f"Genie warehouse: {genie_warehouse_id}")

    created = []
    for agent in AGENTS:
        payload = json.dumps(build_serialized_space(agent))
        existing = find_space(w, agent["title"])
        if existing:
            space = w.genie.update_space(existing.space_id, serialized_space=payload,
                                         description=agent["description"],
                                         warehouse_id=genie_warehouse_id)
            print(f"Updated Genie agent {agent['title']!r}: {space.space_id}")
        else:
            space = w.genie.create_space(warehouse_id=genie_warehouse_id,
                                         serialized_space=payload,
                                         title=agent["title"], description=agent["description"],
                                         parent_path=args.parent_path)
            print(f"Created Genie agent {agent['title']!r}: {space.space_id}")
        print(f"  {w.config.host}/genie/rooms/{space.space_id}")
        created.append((agent["key"], agent["title"], space.space_id))

    if not args.no_bundle:
        upsert_bundle_resources(created)


if __name__ == "__main__":
    main()
