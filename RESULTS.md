# Results (synthetic data — rerun on real data before quoting these numbers)

## Business analytics
| Metric | Value |
|---|---|
| Store-SKU-days analysed | 262,560 (6 stores × 80 SKUs) |
| GMV | ₹43.1 Cr |
| Stockout rate | 6.3% of SKU-days |
| Estimated GMV lost to stockouts | ₹20.6 lakh |
| A-class SKUs (80% of revenue) | 43 (54% of catalogue) |
| Highest promo uplift | Snacks: +46% units |
| Rain-day order lift | +14% |

## Forecasting (14-day horizon, 3 rolling-origin backtests)
| Model | WMAPE |
|---|---|
| Seasonal naive | 40.3% |
| 28-day moving average | 30.3% |
| LightGBM without promo/festival drivers | 29.0% |
| **LightGBM, full** | **28.1%** (bias -0.1%) |

Promo-day WMAPE: 32.8% without driver features → 27.8% with them.

## Replenishment simulation (28 held-out days, lead time 1 day, ABC service targets)
- **Same stock held:** stockout SKU-days 3.56% → 1.32% (**−63%**), lost units −75%
- **Same fill rate (99.1%):** overnight stock 1.01 → 0.70 days of cover (**−31%**)

## Resume bullets (auto-generated from the numbers above)
- Built an end-to-end demand forecasting system for quick-commerce dark stores on **263K+ store-SKU-days**, with DuckDB SQL analytics for ABC–XYZ segmentation, promo uplift, festival/weather effects and stockout-driven lost GMV (₹21L).
- Trained a global LightGBM (Tweedie) model with lag, rolling, calendar, promo and festival features and censored-demand correction; **cut 14-day WMAPE by 30%** (40% → 28%) over a seasonal-naive baseline across 3 rolling backtests.
- Converted forecasts into an ABC-tiered order-up-to replenishment policy; simulation showed **63% fewer stockouts at equal inventory**, or **31% less stock at the same 99% fill rate**.
- Shipped a Streamlit dashboard for GMV/stockout KPIs, SKU-level forecast drill-down and a service-level-driven replenishment planner.
