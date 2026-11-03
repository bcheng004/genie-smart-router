"""Genie definitions for the fictional Asana revenue and marketing datasets."""
from scripts.generate_asana_data import qualified_schema


def build_asana_agents(catalog: str, schema: str, column, claim_views: bool = False) -> list[dict]:
    namespace = qualified_schema(catalog, schema)
    revenue = f"{namespace}.revenue_by_claim" if claim_views else f"{namespace}.revenue"
    marketing = f"{namespace}.marketing_by_claim" if claim_views else f"{namespace}.marketing"
    common = [
        "All data is fully synthetic for an Asana-style SaaS demo. Never present it as "
        "Asana's actual results, customers, prices or guidance. Clearly label every answer synthetic.",
        "All monetary values are USD. Dates cover November 2025 through October 2026 in the "
        "default generation run; inspect actual date bounds if the generator parameters changed.",
        "Do not join campaign-day marketing facts directly to account-month revenue facts. "
        "Aggregate each side to campaign and month before any comparison to avoid fan-out.",
    ]
    if claim_views:
        common.append("Query only the configured _by_claim views, never the underlying base tables. "
                      "The server-issued OAuth claim controls revenue visibility. A user without "
                      "the admin claim cannot access revenue; NULL attributed MRR is restricted, "
                      "not zero. Never infer, reconstruct or invent restricted values.")
    return [
        {
            "key": "asana_revenue_genie",
            "title": "Asana Revenue (Synthetic Claims Router)" if claim_views else "Asana Revenue (Synthetic Router)",
            "description": "Fully synthetic Asana-style SaaS revenue: monthly subscription revenue, "
                           "MRR, ARR, seats, plans, regions, acquisition channels, expansion and churn. "
                           "Not real Asana financial or customer data.",
            "tables": {revenue: [
                column("billing_month", "Snapshot month, first calendar day; grain is account_id and billing_month."),
                column("account_id", "Fictional account identifier. Appears once per month."),
                column("account_name", "Faker-generated fictional company; not a real Asana customer."),
                column("plan", "Invented SaaS tier: Starter, Advanced or Enterprise.", match=True),
                column("region", "Acquisition campaign region: North America, EMEA or APAC.", match=True),
                column("channel", "Single-touch acquisition channel.", ["acquisition channel"], match=True),
                column("campaign_id", "Synthetic acquisition campaign identifier; not a unique revenue key."),
                column("acquisition_date", "Account signup date; may predate the snapshot period."),
                column("status", "Active, Churned or Not yet acquired. Inactive rows have zero seats and MRR.", match=True),
                column("seats", "Active paid seats at month-end; a snapshot, not additive over time."),
                column("opening_mrr_usd", "Monthly recurring revenue at the start of this month, USD."),
                column("new_mrr_usd", "MRR first added by accounts acquired during this month, USD."),
                column("expansion_mrr_usd", "Incremental MRR from seat expansion this month, USD."),
                column("churn_mrr_usd", "Positive MRR lost to churn this month, USD; subtract in an MRR bridge."),
                column("mrr_usd", "Closing monthly recurring revenue, USD. Sum across accounts, never across months.", ["MRR"]),
                column("arr_usd", "Closing annual recurring revenue, USD, equal to closing MRR times 12. "
                       "Never sum snapshots across months.", ["ARR"]),
                column("subscription_revenue_usd", "Simplified monthly subscription billings equal to closing MRR, "
                       "without proration; not GAAP recognized revenue. Additive across months.", ["revenue", "billings"]),
                column("is_synthetic", "Always true; no actual Asana data.", exclude=True),
            ]},
            "joins": [],
            "measures": [
                {"alias": "closing_mrr", "display_name": "Closing MRR (USD)",
                 "sql": "SUM(revenue.mrr_usd)", "validate_from": f"{revenue} AS revenue"},
                {"alias": "subscription_revenue", "display_name": "Synthetic subscription revenue (USD)",
                 "sql": "SUM(revenue.subscription_revenue_usd)", "validate_from": f"{revenue} AS revenue"},
            ],
            "filters": [],
            "expressions": [],
            "examples": [
                ("Show synthetic monthly revenue and closing MRR.",
                 f"SELECT revenue.billing_month, SUM(revenue.subscription_revenue_usd) AS subscription_revenue_usd, "
                 f"SUM(revenue.mrr_usd) AS closing_mrr_usd FROM {revenue} AS revenue "
                 "GROUP BY revenue.billing_month ORDER BY revenue.billing_month"),
                ("What is the latest synthetic ARR by plan?",
                 f"SELECT revenue.plan, SUM(revenue.arr_usd) AS arr_usd FROM {revenue} AS revenue "
                 f"WHERE revenue.billing_month = (SELECT MAX(latest.billing_month) FROM {revenue} AS latest) "
                 "GROUP BY revenue.plan ORDER BY arr_usd DESC"),
                ("Show the monthly synthetic MRR bridge.",
                 f"SELECT revenue.billing_month, SUM(revenue.opening_mrr_usd) AS opening_mrr_usd, "
                 "SUM(revenue.new_mrr_usd) AS new_mrr_usd, SUM(revenue.expansion_mrr_usd) AS expansion_mrr_usd, "
                 "SUM(revenue.churn_mrr_usd) AS churn_mrr_usd, SUM(revenue.mrr_usd) AS closing_mrr_usd "
                 f"FROM {revenue} AS revenue GROUP BY revenue.billing_month ORDER BY revenue.billing_month"),
                ("Which acquisition channels added the most synthetic new MRR by month?",
                 f"SELECT revenue.billing_month, revenue.channel, SUM(revenue.new_mrr_usd) AS new_mrr_usd "
                 f"FROM {revenue} AS revenue GROUP BY revenue.billing_month, revenue.channel "
                 "ORDER BY revenue.billing_month, new_mrr_usd DESC"),
            ],
            "sample_questions": [
                "Show synthetic monthly subscription revenue.",
                "What is the latest ARR by plan and region?",
                "Show the monthly MRR bridge: opening, new, expansion, churn and closing.",
                "How did new MRR from Paid Social change in July and August 2026?",
            ],
            "text_instructions": common + [
                "MRR bridge: closing = opening + new + expansion - churn. There is no contraction in this demo.",
                "Use the latest billing_month for current MRR, ARR, seats or active-account questions. "
                "For monthly trends group by billing_month. For revenue over a period sum subscription_revenue_usd.",
            ],
        },
        {
            "key": "asana_marketing_genie",
            "title": "Asana Marketing (Synthetic Claims Router)" if claim_views else "Asana Marketing (Synthetic Router)",
            "description": "Fully synthetic Asana-style campaign performance: marketing spend, channels, "
                           "lead funnel, customer acquisition costs and attributed new MRR. "
                           "Investigate a July-August 2026 paid-social lead-quality decline; not real Asana data.",
            "tables": {marketing: [
                column("event_date", "Campaign activity date; grain is campaign_id and event_date."),
                column("campaign_id", "Synthetic campaign identifier, repeated on each activity date."),
                column("campaign_name", "Fictional campaign name."),
                column("channel", "Paid Search, Paid Social, Organic, Webinar or Partner.", match=True),
                column("region", "Campaign region: North America, EMEA or APAC.", match=True),
                column("scenario", "Baseline or a simulated paid-social lead-quality decline.", match=True),
                column("spend_usd", "Daily marketing cost, USD. Includes channel operating costs, not only paid media.", ["spend", "cost"]),
                column("impressions", "Synthetic daily campaign reach impressions."),
                column("clicks", "Daily clicks; never fewer than leads."),
                column("leads", "Daily generated leads."),
                column("qualified_leads", "Daily marketing-qualified leads, no greater than leads.", ["MQLs"]),
                column("opportunities", "Daily sales opportunities, no greater than qualified leads."),
                column("converted_accounts", "New paying accounts acquired on this day, reconciled to revenue.", ["conversions", "new customers"]),
                column("attributed_new_mrr_usd", "Single-touch new subscription MRR attributable to conversions "
                       "on this date, USD. Reconciles to revenue.new_mrr_usd by campaign and month; "
                       "not total revenue or lifetime value.", ["attributed MRR"]),
                column("is_synthetic", "Always true; no actual Asana data.", exclude=True),
            ]},
            "joins": [],
            "measures": [
                {"alias": "marketing_spend", "display_name": "Marketing spend (USD)",
                 "sql": "SUM(marketing.spend_usd)", "validate_from": f"{marketing} AS marketing"},
                {"alias": "cost_per_acquisition", "display_name": "Cost per acquisition (USD)",
                 "sql": "SUM(marketing.spend_usd) / NULLIF(SUM(marketing.converted_accounts), 0)",
                 "validate_from": f"{marketing} AS marketing"},
            ],
            "filters": [],
            "expressions": [],
            "examples": [
                ("Show synthetic monthly marketing performance by channel.",
                 "SELECT DATE_TRUNC('month', marketing.event_date) AS activity_month, marketing.channel, "
                 "SUM(marketing.spend_usd) AS spend_usd, SUM(marketing.leads) AS leads, "
                 "SUM(marketing.converted_accounts) AS conversions, "
                 "SUM(marketing.attributed_new_mrr_usd) AS attributed_new_mrr_usd "
                 f"FROM {marketing} AS marketing GROUP BY DATE_TRUNC('month', marketing.event_date), "
                 "marketing.channel ORDER BY activity_month, marketing.channel"),
                ("Why did synthetic paid-social acquisition costs increase?",
                 "SELECT DATE_TRUNC('month', marketing.event_date) AS activity_month, "
                 "SUM(marketing.spend_usd) AS spend_usd, SUM(marketing.converted_accounts) AS conversions, "
                 "SUM(marketing.spend_usd) / NULLIF(SUM(marketing.converted_accounts), 0) AS acquisition_cost_usd, "
                 "SUM(marketing.qualified_leads) / NULLIF(SUM(marketing.leads), 0) AS qualified_lead_rate "
                 f"FROM {marketing} AS marketing WHERE marketing.channel = 'Paid Social' "
                 "GROUP BY DATE_TRUNC('month', marketing.event_date) ORDER BY activity_month"),
                ("Which synthetic campaigns generated the most attributed new MRR per dollar spent?",
                 "SELECT marketing.campaign_name, SUM(marketing.spend_usd) AS spend_usd, "
                 "SUM(marketing.attributed_new_mrr_usd) / NULLIF(SUM(marketing.spend_usd), 0) AS new_mrr_per_spend "
                 f"FROM {marketing} AS marketing GROUP BY marketing.campaign_name "
                 "ORDER BY new_mrr_per_spend DESC LIMIT 10"),
            ],
            "sample_questions": [
                "Compare monthly marketing spend and new paying accounts by channel.",
                "Why did Paid Social acquisition costs rise in July and August 2026?",
                "Which campaigns generated the most attributed new MRR per dollar spent?",
                "Show the conversion funnel by region.",
            ],
            "text_instructions": common + [
                "Compute rates as ratios of sums, never averages of row-level ratios. Use NULLIF for zero denominators.",
                "Cost per acquisition is spend divided by converted_accounts. Attributed new MRR divided by spend "
                "is an MRR-to-spend ratio, not realized ROAS, profit ROI or lifetime-value return.",
                "The default demo deliberately raises Paid Social spending and lowers lead quality in July-August "
                "2026; some acquisitions shift to September-October. Check observed results rather than inventing causality.",
            ],
        },
    ]
