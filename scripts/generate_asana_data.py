"""Generate fictional Asana-style SaaS data with Databricks Connect serverless."""
import argparse
from datetime import date, timedelta
import re


def qualified_schema(catalog: str, schema: str) -> str:
    if not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part) for part in (catalog, schema)):
        raise ValueError("Catalog and schema must contain only letters, numbers and underscores")
    return f"{catalog}.{schema}"


def month_start(start: date, offset: int) -> date:
    total = start.year * 12 + start.month - 1 + offset
    return date(total // 12, total % 12 + 1, 1)


def validate_data(spark, namespace: str) -> None:
    revenue = f"{namespace}.revenue"
    marketing = f"{namespace}.marketing"
    checks = {
        "revenue MRR bridge and synthetic flags": f"""
            SELECT COUNT(*) AS failures FROM {revenue}
            WHERE NOT is_synthetic OR mrr_usd < 0 OR seats < 0
               OR mrr_usd <> opening_mrr_usd + new_mrr_usd + expansion_mrr_usd - churn_mrr_usd
               OR arr_usd <> mrr_usd * 12 OR subscription_revenue_usd <> mrr_usd
        """,
        "unique revenue grain": f"""
            SELECT COUNT(*) AS failures FROM (
              SELECT account_id, billing_month FROM {revenue}
              GROUP BY account_id, billing_month HAVING COUNT(*) <> 1)
        """,
        "marketing funnel and synthetic flags": f"""
            SELECT COUNT(*) AS failures FROM {marketing}
            WHERE NOT is_synthetic OR spend_usd < 0 OR converted_accounts < 0
               OR impressions < clicks OR clicks < leads OR leads < qualified_leads
               OR qualified_leads < opportunities OR opportunities < converted_accounts
        """,
        "unique marketing grain": f"""
            SELECT COUNT(*) AS failures FROM (
              SELECT campaign_id, event_date FROM {marketing}
              GROUP BY campaign_id, event_date HAVING COUNT(*) <> 1)
        """,
        "campaign-month conversion and MRR reconciliation": f"""
            WITH revenue_cohorts AS (
              SELECT campaign_id, billing_month, COUNT(*) AS conversions, SUM(new_mrr_usd) AS new_mrr
              FROM {revenue} WHERE new_mrr_usd > 0 GROUP BY campaign_id, billing_month
            ), marketing_cohorts AS (
              SELECT campaign_id, CAST(DATE_TRUNC('month', event_date) AS DATE) AS billing_month,
                     SUM(converted_accounts) AS conversions, SUM(attributed_new_mrr_usd) AS new_mrr
              FROM {marketing} GROUP BY campaign_id, CAST(DATE_TRUNC('month', event_date) AS DATE)
            )
            SELECT COUNT(*) AS failures FROM revenue_cohorts AS revenue
            FULL OUTER JOIN marketing_cohorts AS marketing
              ON revenue.campaign_id = marketing.campaign_id AND revenue.billing_month = marketing.billing_month
            WHERE COALESCE(revenue.conversions, 0) <> COALESCE(marketing.conversions, 0)
               OR COALESCE(revenue.new_mrr, 0) <> COALESCE(marketing.new_mrr, 0)
        """,
    }
    for label, statement in checks.items():
        failures = spark.sql(statement).first()["failures"]
        if failures:
            raise ValueError(f"Data validation failed: {label}: {failures} rows")
        print(f"Validated {label}", flush=True)


def generate(spark, namespace: str, start: date, months: int, accounts: int,
             campaigns: int, overwrite: bool = False) -> None:
    from pyspark.sql import functions as functions
    import pandas as pd

    end = month_start(start, months) - timedelta(days=1)
    days = (end - start).days + 1
    revenue_table = f"{namespace}.revenue"
    marketing_table = f"{namespace}.marketing"
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {namespace} COMMENT 'Fully synthetic Asana demo data'")
    if not overwrite and any(spark.catalog.tableExists(table)
                             for table in (revenue_table, marketing_table)):
        raise ValueError("Demo tables already exist; pass --overwrite to explicitly replace them")

    def bucket(salt, modulus):
        return functions.pmod(functions.xxhash64("id", functions.lit(salt)), functions.lit(modulus))

    def channel(campaign):
        category = functions.pmod(campaign, functions.lit(20))
        return (functions.when(category < 8, "Paid Search")
                .when(category < 13, "Paid Social").when(category < 17, "Organic")
                .when(category < 19, "Webinar").otherwise("Partner"))

    def region(campaign):
        category = functions.pmod(campaign, functions.lit(100))
        return (functions.when(category < 55, "North America")
                .when(category < 85, "EMEA").otherwise("APAC"))

    @functions.pandas_udf("string")
    def synthetic_names(identifiers: pd.Series) -> pd.Series:
        from faker import Faker
        fake = Faker("en_US")

        def company(identifier):
            fake.seed_instance(int(identifier))
            return fake.company() + " (Synthetic)"

        return identifiers.apply(company)

    base = (spark.range(accounts, numPartitions=8)
            .withColumn("account_id", functions.format_string("SYN-ACCT-%06d", functions.col("id")))
            .withColumn("account_name", synthetic_names("id"))
            .withColumn("campaign_number", bucket("campaign", campaigns).cast("int"))
            .withColumn("campaign_id", functions.format_string("SYN-CAMP-%03d", functions.col("campaign_number")))
            .withColumn("channel", channel(functions.col("campaign_number")))
            .withColumn("region", region(functions.col("campaign_number")))
            .withColumn("plan", functions.when(bucket("plan", 100) < 55, "Starter")
                        .when(bucket("plan", 100) < 85, "Advanced").otherwise("Enterprise"))
            .withColumn("base_seats", functions.greatest(functions.lit(3), functions.round(
                functions.exp(functions.randn(42) * 0.65 + functions.when(
                    functions.col("plan") == "Starter", 2.5).when(
                    functions.col("plan") == "Advanced", 3.8).otherwise(5.3)))).cast("int"))
            .withColumn("seat_price", functions.when(functions.col("plan") == "Starter", 14)
                        .when(functions.col("plan") == "Advanced", 29).otherwise(49))
            .withColumn("discount", bucket("discount", 16) / 100.0)
            .withColumn("acquisition_date", functions.when(bucket("existing", 100) < 60,
                        functions.date_sub(functions.lit(start), (bucket("history", 730) + 1).cast("int")))
                        .otherwise(functions.date_add(functions.lit(start), bucket("signup", days).cast("int")))))
    anomaly_start = month_start(start, 8)
    anomaly_end = month_start(start, 10) - timedelta(days=1)
    base = (base.withColumn("acquisition_date", functions.when(
                (functions.col("channel") == "Paid Social")
                & functions.col("acquisition_date").between(anomaly_start, anomaly_end)
                & (bucket("quality_loss", 100) < 65),
                functions.least(functions.lit(end), functions.date_add("acquisition_date", 61)))
                .otherwise(functions.col("acquisition_date")))
            .withColumn("activation_month", functions.months_between(
                functions.trunc("acquisition_date", "month"), functions.lit(start)).cast("int"))
            .withColumn("expansion_month", functions.greatest(functions.lit(0), functions.col("activation_month"))
                        + 2 + bucket("expansion_timing", 8))
            .withColumn("churn_month", functions.greatest(functions.lit(0), functions.col("activation_month"))
                        + 3 + bucket("churn_timing", 9))
            .withColumn("expands", bucket("expands", 100) < 24)
            .withColumn("churns", bucket("churns", 100) < 9))

    def seats_at(month_index):
        active = ((month_index >= functions.col("activation_month"))
                  & (~functions.col("churns") | (month_index < functions.col("churn_month"))))
        expanded = functions.col("expands") & (month_index >= functions.col("expansion_month"))
        return functions.when(active, functions.when(expanded, functions.ceil(
            functions.col("base_seats") * 1.2)).otherwise(functions.col("base_seats"))).otherwise(0).cast("int")

    def mrr_at(month_index):
        return functions.round(seats_at(month_index) * functions.col("seat_price")
                               * (1 - functions.col("discount")), 2).cast("decimal(18,2)")

    snapshots = base.crossJoin(spark.range(months).withColumnRenamed("id", "month_index"))
    current = functions.col("month_index")
    snapshots = (snapshots.withColumn("billing_month", functions.add_months(functions.lit(start), current.cast("int")))
                 .withColumn("seats", seats_at(current))
                 .withColumn("opening_mrr_usd", mrr_at(current - 1))
                 .withColumn("mrr_usd", mrr_at(current))
                 .withColumn("new_mrr_usd", functions.when(current == functions.col("activation_month"),
                             functions.col("mrr_usd")).otherwise(0).cast("decimal(18,2)"))
                 .withColumn("expansion_mrr_usd", functions.greatest(functions.lit(0),
                             functions.col("mrr_usd") - functions.col("opening_mrr_usd")
                             - functions.col("new_mrr_usd")).cast("decimal(18,2)"))
                 .withColumn("churn_mrr_usd", functions.greatest(functions.lit(0),
                             functions.col("opening_mrr_usd") - functions.col("mrr_usd")).cast("decimal(18,2)"))
                 .withColumn("subscription_revenue_usd", functions.col("mrr_usd"))
                 .withColumn("arr_usd", (functions.col("mrr_usd") * 12).cast("decimal(18,2)"))
                 .withColumn("status", functions.when(current < functions.col("activation_month"), "Not yet acquired")
                             .when(functions.col("mrr_usd") > 0, "Active").otherwise("Churned"))
                 .withColumn("is_synthetic", functions.lit(True)))
    mode = "overwrite" if overwrite else "error"
    revenue_columns = ["billing_month", "account_id", "account_name", "plan", "region", "channel",
                       "campaign_id", "acquisition_date", "status", "seats", "opening_mrr_usd",
                       "new_mrr_usd", "expansion_mrr_usd", "churn_mrr_usd", "mrr_usd", "arr_usd",
                       "subscription_revenue_usd", "is_synthetic"]
    snapshots.select(*revenue_columns).write.format("delta").mode(mode).saveAsTable(revenue_table)
    spark.sql(f"COMMENT ON TABLE {revenue_table} IS 'SYNTHETIC ONLY: fictional Asana-style SaaS accounts; "
              "one row per account per month, including inactive accounts. USD prices are invented. "
              "Monthly subscription revenue equals closing MRR with no proration; not GAAP revenue. "
              "MRR and ARR are snapshots: never sum across months.'")
    print(f"Created {revenue_table}: {accounts * months:,} account-month rows", flush=True)

    conversions = (spark.table(revenue_table).filter("new_mrr_usd > 0")
                   .groupBy("campaign_id", "acquisition_date")
                   .agg(functions.count("account_id").alias("converted_accounts"),
                        functions.sum("new_mrr_usd").alias("attributed_new_mrr_usd"))
                   .withColumnRenamed("acquisition_date", "event_date"))
    campaign_data = (spark.range(campaigns, numPartitions=8)
                     .withColumn("campaign_id", functions.format_string("SYN-CAMP-%03d", functions.col("id")))
                     .withColumn("channel", channel(functions.col("id")))
                     .withColumn("region", region(functions.col("id")))
                     .withColumn("campaign_name", functions.concat(functions.lit("Synthetic "),
                                 functions.col("channel"), functions.lit(" "), functions.col("campaign_id")))
                     .withColumn("daily_budget", functions.exp(functions.randn(43) * 0.6 + 4.5)))
    dates = spark.range(days).select(functions.date_add(functions.lit(start), functions.col("id").cast("int"))
                                    .alias("event_date"))
    marketing = campaign_data.crossJoin(dates).join(conversions, ["campaign_id", "event_date"], "left")
    anomaly = ((functions.col("channel") == "Paid Social")
               & functions.col("event_date").between(anomaly_start, anomaly_end))
    marketing = (marketing.fillna({"converted_accounts": 0, "attributed_new_mrr_usd": 0})
                 .withColumn("scenario", functions.when(anomaly, "Paid social lead-quality decline")
                             .otherwise("Baseline"))
                 .withColumn("spend_usd", functions.round(functions.col("daily_budget")
                             * functions.when(anomaly, 1.65).otherwise(1.0)
                             * functions.when(functions.dayofweek("event_date").isin(1, 7), 0.7).otherwise(1.0)
                             * (0.85 + functions.pmod(functions.xxhash64("campaign_id", "event_date"),
                                                       functions.lit(31)) / 100.0), 2).cast("decimal(18,2)"))
                 .withColumn("leads", functions.greatest(functions.col("converted_accounts") * 10,
                             functions.round(functions.col("spend_usd") / functions.when(anomaly, 18.0)
                                             .otherwise(9.0))).cast("long"))
                 .withColumn("qualified_leads", functions.greatest(functions.col("converted_accounts"),
                             functions.floor(functions.col("leads")
                                             * functions.when(anomaly, 0.18).otherwise(0.45))))
                 .withColumn("opportunities", functions.greatest(functions.col("converted_accounts"),
                             functions.floor(functions.col("qualified_leads") * 0.4)))
                 .withColumn("clicks", (functions.col("leads") * 7).cast("long"))
                 .withColumn("impressions", (functions.col("clicks") * 40).cast("long"))
                 .withColumn("attributed_new_mrr_usd", functions.col("attributed_new_mrr_usd").cast("decimal(18,2)"))
                 .withColumn("is_synthetic", functions.lit(True)))
    marketing.select("event_date", "campaign_id", "campaign_name", "channel", "region", "scenario",
                     "spend_usd", "impressions", "clicks", "leads", "qualified_leads", "opportunities",
                     "converted_accounts", "attributed_new_mrr_usd", "is_synthetic") \
        .write.format("delta").mode(mode).saveAsTable(marketing_table)
    spark.sql(f"COMMENT ON TABLE {marketing_table} IS 'SYNTHETIC ONLY: fictional campaign-day marketing "
              "funnel. Conversions and attributed new MRR reconcile to revenue acquisition cohorts. "
              "Single-touch campaign attribution; spend includes non-ad channel operating costs. "
              "Never join campaign-day rows directly to account-month rows: aggregate first.'")
    print(f"Created {marketing_table}: {campaigns * days:,} campaign-day rows", flush=True)
    validate_data(spark, namespace)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--start-month", type=date.fromisoformat, default=date(2025, 11, 1))
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--accounts", type=int, default=10000)
    parser.add_argument("--campaigns", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    namespace = qualified_schema(args.catalog, args.schema)
    if args.start_month.day != 1 or args.months < 12 or args.accounts < 1 or args.campaigns < 20:
        parser.error("Use a month-start date, at least 12 months, positive accounts and at least 20 campaigns")
    from databricks.connect import DatabricksSession
    from databricks.connect.env import DatabricksEnv
    from importlib.metadata import version
    environment = DatabricksEnv().withDependencies(f"Faker=={version('Faker')}")
    spark = (DatabricksSession.builder.profile(args.profile).serverless(True)
             .withEnvironment(environment).getOrCreate())
    try:
        generate(spark, namespace, args.start_month, args.months, args.accounts, args.campaigns, args.overwrite)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
