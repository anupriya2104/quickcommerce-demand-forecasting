-- =====================================================================
-- Quick-commerce demand analytics (DuckDB / ANSI SQL, Postgres-compatible)
-- Each block is executed by src/analytics.py; '-- name:' tags the result.
-- =====================================================================

-- name: kpi_overview
SELECT
    COUNT(DISTINCT store_id)                         AS stores,
    COUNT(DISTINCT sku_id)                           AS skus,
    MIN(date)                                        AS start_date,
    MAX(date)                                        AS end_date,
    SUM(units_sold)                                  AS units_sold,
    ROUND(SUM(units_sold * selling_price) / 1e7, 2)  AS gmv_crore,
    ROUND(AVG(stockout_flag) * 100, 2)               AS stockout_rate_pct,
    ROUND(AVG(on_promo) * 100, 2)                    AS promo_day_share_pct
FROM sales;

-- name: monthly_gmv_by_category
SELECT
    DATE_TRUNC('month', s.date)                     AS month,
    p.category,
    SUM(s.units_sold)                               AS units,
    ROUND(SUM(s.units_sold * s.selling_price), 0)   AS gmv
FROM sales s JOIN products p USING (sku_id)
GROUP BY 1, 2
ORDER BY 1, 2;

-- name: category_yoy_growth
-- Same-window comparison: the latest 182 days vs the same 182 days one year earlier,
-- so seasonality does not inflate growth. Works for any date range >= 1.5 years.
WITH bounds AS (SELECT MAX(date) AS d1 FROM sales),
w AS (
    SELECT p.category,
           SUM(s.units_sold * s.selling_price) FILTER (WHERE s.date >  b.d1 - INTERVAL 182 DAY)                 AS gmv_recent,
           SUM(s.units_sold * s.selling_price) FILTER (WHERE s.date >  b.d1 - INTERVAL 547 DAY
                                                         AND s.date <= b.d1 - INTERVAL 365 DAY)                 AS gmv_prior_year
    FROM sales s JOIN products p USING (sku_id), bounds b
    GROUP BY p.category
)
SELECT category, ROUND(gmv_prior_year, 0) AS gmv_prior_year, ROUND(gmv_recent, 0) AS gmv_recent,
       ROUND(100.0 * (gmv_recent / gmv_prior_year - 1), 1) AS yoy_growth_pct
FROM w
ORDER BY yoy_growth_pct DESC;

-- name: abc_xyz
-- ABC: cumulative revenue share (A = top 80%, B = next 15%, C = last 5%).
-- XYZ: coefficient of variation of weekly store-level demand (X < 0.25 stable, Y < 0.30, Z volatile).
WITH sku_rev AS (
    SELECT sku_id, SUM(units_sold * selling_price) AS revenue
    FROM sales GROUP BY sku_id
),
abc AS (
    SELECT sku_id, revenue,
           SUM(revenue) OVER (ORDER BY revenue DESC ROWS UNBOUNDED PRECEDING)
             / SUM(revenue) OVER () AS cum_share
    FROM sku_rev
),
weekly AS (   -- store x SKU x week: the level where replenishment decisions are made
    SELECT store_id, sku_id, DATE_TRUNC('week', date) AS wk, SUM(units_sold) AS units
    FROM sales WHERE date > (SELECT MAX(date) FROM sales) - INTERVAL 182 DAY GROUP BY 1, 2, 3   -- recent 6 months limits trend effect
),
xyz AS (
    SELECT sku_id, AVG(cv) AS cv FROM (
        SELECT store_id, sku_id, STDDEV_SAMP(units) / AVG(units) AS cv FROM weekly GROUP BY 1, 2
    ) GROUP BY sku_id
)
SELECT
    a.sku_id, p.category, ROUND(a.revenue, 0) AS revenue, ROUND(a.cum_share, 4) AS cum_share,
    CASE WHEN a.cum_share <= 0.80 THEN 'A' WHEN a.cum_share <= 0.95 THEN 'B' ELSE 'C' END AS abc_class,
    ROUND(x.cv, 3) AS weekly_cv,
    CASE WHEN x.cv < 0.25 THEN 'X' WHEN x.cv < 0.30 THEN 'Y' ELSE 'Z' END AS xyz_class
FROM abc a JOIN xyz x USING (sku_id) JOIN products p USING (sku_id)
ORDER BY a.revenue DESC;

-- name: dow_index
-- Day-of-week demand index per category (100 = category's average day).
WITH d AS (
    SELECT p.category, DAYOFWEEK(s.date) AS dow, s.date, SUM(s.units_sold) AS units
    FROM sales s JOIN products p USING (sku_id)
    WHERE s.on_promo = 0 AND s.stockout_flag = 0
    GROUP BY 1, 2, 3
)
SELECT category, dow,
       ROUND(100.0 * AVG(units) / AVG(AVG(units)) OVER (PARTITION BY category), 1) AS demand_index
FROM d
GROUP BY category, dow
ORDER BY category, dow;

-- name: promo_uplift
-- Uplift = promo-day units vs the same SKU-store's non-promo days in the same month
-- (controls for seasonality and store size). Stockout days excluded (censored).
WITH base AS (
    SELECT store_id, sku_id, DATE_TRUNC('month', date) AS m,
           AVG(units_sold) FILTER (WHERE on_promo = 0) AS base_units,
           AVG(units_sold) FILTER (WHERE on_promo = 1) AS promo_units,
           AVG(discount_pct) FILTER (WHERE on_promo = 1) AS avg_disc
    FROM sales
    WHERE stockout_flag = 0
    GROUP BY 1, 2, 3
    HAVING COUNT(*) FILTER (WHERE on_promo = 1) > 0 AND COUNT(*) FILTER (WHERE on_promo = 0) >= 10
)
SELECT p.category,
       COUNT(*)                                            AS promo_events,
       ROUND(AVG(avg_disc) * 100, 1)                       AS avg_discount_pct,
       ROUND(100.0 * (AVG(promo_units) / AVG(base_units) - 1), 1) AS uplift_pct,
       ROUND((AVG(promo_units) / AVG(base_units) - 1) / AVG(avg_disc), 2) AS uplift_per_unit_discount
FROM base b JOIN products p USING (sku_id)
GROUP BY p.category
ORDER BY uplift_pct DESC;

-- name: festival_lift
WITH daily AS (
    SELECT s.date, p.category, SUM(s.units_sold) AS units
    FROM sales s JOIN products p USING (sku_id)
    GROUP BY 1, 2
),
tagged AS (
    SELECT d.*, c.festival_name,
           AVG(units) OVER (PARTITION BY category ORDER BY d.date
                            ROWS BETWEEN 28 PRECEDING AND 7 PRECEDING) AS trailing_base
    FROM daily d JOIN calendar c USING (date)
)
SELECT festival_name, category, ROUND(100.0 * (AVG(units / trailing_base) - 1), 1) AS lift_pct
FROM tagged
WHERE festival_name IS NOT NULL AND trailing_base IS NOT NULL
GROUP BY 1, 2
ORDER BY 1, lift_pct DESC;

-- name: rain_effect
SELECT CASE WHEN c.rain_mm > 5 THEN 'Rain day (>5mm)' ELSE 'Dry day' END AS weather,
       COUNT(DISTINCT s.date)                               AS days,
       ROUND(SUM(s.units_sold) * 1.0 / COUNT(DISTINCT s.date), 0) AS units_per_day
FROM sales s JOIN calendar c USING (date)
WHERE c.is_festival = 0
GROUP BY 1;

-- name: stockout_by_store
-- Lost sales estimate: on stockout days, demand is at least the SKU-store's
-- trailing 14-day average of in-stock days; lost = max(0, that - sold).
WITH s2 AS (
    SELECT *,
           AVG(CASE WHEN stockout_flag = 0 THEN units_sold END)
               OVER (PARTITION BY store_id, sku_id ORDER BY date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS exp_units
    FROM sales
)
SELECT st.locality AS store,
       ROUND(AVG(stockout_flag) * 100, 2)                                         AS stockout_rate_pct,
       ROUND(SUM(CASE WHEN stockout_flag = 1 THEN GREATEST(exp_units - units_sold, 0) ELSE 0 END), 0) AS est_lost_units,
       ROUND(SUM(CASE WHEN stockout_flag = 1 THEN GREATEST(exp_units - units_sold, 0) * selling_price ELSE 0 END) / 1e5, 2) AS est_lost_gmv_lakh
FROM s2 JOIN stores st USING (store_id)
GROUP BY st.locality
ORDER BY est_lost_gmv_lakh DESC;
