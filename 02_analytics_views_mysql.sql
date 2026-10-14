-- ============================================================================
-- Analytics views (MySQL 8) — the SQL layer of the Data Analytics skill.
-- The Python pipeline computes the same numbers with pandas; these views are
-- for analysts/BI tools querying the production database directly.
-- ============================================================================
USE card_intelligence;

-- 1. Monthly portfolio KPIs ------------------------------------------------
CREATE OR REPLACE VIEW v_monthly_kpis AS
SELECT
    e.month,
    COUNT(DISTINCT e.cardholder_id)              AS active_cardholders,
    ROUND(AVG(e.engagement_score), 2)            AS avg_engagement_score,
    ROUND(SUM(e.spend), 2)                       AS total_spend,
    SUM(e.logins)                                AS total_logins,
    SUM(e.email_clicks)                          AS total_email_clicks,
    SUM(CASE WHEN e.engagement_decline = 1 THEN 1 ELSE 0 END) AS decline_flags
FROM engagement_monthly e
GROUP BY e.month;

-- 2. Cardholder RFM + engagement (segmentation input) -----------------------
CREATE OR REPLACE VIEW v_cardholder_rfm AS
SELECT
    c.cardholder_id,
    c.segment,
    c.industry,
    c.region,
    DATEDIFF((SELECT MAX(txn_date) FROM transactions), MAX(t.txn_date)) AS recency_days,
    COUNT(t.txn_id)                                                          AS frequency,
    ROUND(SUM(t.amount), 2)                                                  AS monetary,
    ROUND(AVG(e.engagement_score), 2)                                        AS avg_engagement,
    ROUND(AVG(e.spend), 2)                                                   AS avg_monthly_spend
FROM cardholders c
JOIN transactions t  ON t.cardholder_id = c.cardholder_id
JOIN engagement_monthly e ON e.cardholder_id = c.cardholder_id
GROUP BY c.cardholder_id, c.segment, c.industry, c.region;

-- 3. Campaign performance ---------------------------------------------------
CREATE OR REPLACE VIEW v_campaign_performance AS
SELECT
    campaign_id,
    campaign_name,
    channel,
    target_segment,
    budget,
    impressions,
    clicks,
    conversions,
    attributed_revenue,
    ROUND(clicks / impressions, 4)                       AS ctr,
    ROUND(conversions / NULLIF(clicks, 0), 4)             AS cvr,
    ROUND(attributed_revenue / budget, 2)                AS roas,
    ROUND(budget / NULLIF(conversions, 0), 2)            AS cpa,
    RANK() OVER (ORDER BY attributed_revenue / budget DESC) AS roas_rank
FROM campaigns;

-- 4. Channel rollup ----------------------------------------------------------
CREATE OR REPLACE VIEW v_channel_rollup AS
SELECT
    channel,
    COUNT(*)                                   AS n_campaigns,
    SUM(budget)                                AS total_budget,
    SUM(attributed_revenue)                    AS total_revenue,
    SUM(conversions)                           AS total_conversions,
    ROUND(SUM(attributed_revenue) / SUM(budget), 2) AS roas
FROM campaigns
GROUP BY channel;

-- 5. Merchant category mix ---------------------------------------------------
CREATE OR REPLACE VIEW v_category_mix AS
SELECT
    merchant_category,
    COUNT(*)                                   AS n_transactions,
    ROUND(SUM(amount), 2)                      AS total_amount,
    ROUND(AVG(amount), 2)                      AS avg_ticket,
    ROUND(SUM(amount) / SUM(SUM(amount)) OVER (), 4) AS share_of_spend
FROM transactions
GROUP BY merchant_category;

-- 6. A/B test summary (two-proportion z-test inputs) -------------------------
CREATE OR REPLACE VIEW v_ab_test_summary AS
SELECT
    arm,
    SUM(impressions)   AS impressions,
    SUM(conversions)   AS conversions,
    ROUND(SUM(conversions) / SUM(impressions), 5) AS cvr
FROM ab_test_daily
GROUP BY arm;

-- 7. Engagement trend with 3-month momentum (decline early-warning) ----------
CREATE OR REPLACE VIEW v_engagement_momentum AS
SELECT
    cardholder_id,
    month,
    engagement_score,
    ROUND(AVG(engagement_score) OVER (
        PARTITION BY cardholder_id ORDER BY month_index
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS score_3m_avg,
    ROUND(engagement_score - AVG(engagement_score) OVER (
        PARTITION BY cardholder_id ORDER BY month_index
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS score_mom
FROM engagement_monthly;
