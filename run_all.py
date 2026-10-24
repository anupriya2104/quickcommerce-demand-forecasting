"""One command end-to-end: SQL analytics -> forecasting backtests -> inventory simulation -> RESULTS.md"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import analytics  # noqa: E402
import forecast  # noqa: E402

t0 = time.time()
print("== Stage 1: SQL analytics ==")
analytics.main()
print("\n== Stage 2: forecasting + replenishment simulation ==")
forecast.main()

a = json.loads((ROOT / "reports" / "analytics_summary.json").read_text(encoding="utf-8"))
m = json.loads((ROOT / "reports" / "model_summary.json").read_text(encoding="utf-8"))
synthetic = (ROOT / "data" / "raw" / ".synthetic").exists()

results = f"""# Results{' (synthetic data — rerun on real data before quoting these numbers)' if synthetic else ''}

## Business analytics
| Metric | Value |
|---|---|
| Store-SKU-days analysed | {a['rows']:,} ({a['stores']} stores × {a['skus']} SKUs) |
| GMV | ₹{a['gmv_crore']:.1f} Cr |
| Stockout rate | {a['stockout_rate_pct']:.1f}% of SKU-days |
| Estimated GMV lost to stockouts | ₹{a['est_lost_gmv_lakh']:.1f} lakh |
| A-class SKUs (80% of revenue) | {a['a_class_skus']} ({a['a_class_sku_share_pct']:.0f}% of catalogue) |
| Highest promo uplift | {a['top_promo_category']}: +{a['top_promo_uplift_pct']:.0f}% units |
| Rain-day order lift | {('+%.0f%%' % a['rain_day_lift_pct']) if a['rain_day_lift_pct'] == a['rain_day_lift_pct'] else 'n/a (no weather data)'} |

## Forecasting (14-day horizon, 3 rolling-origin backtests)
| Model | WMAPE |
|---|---|
| Seasonal naive | {m['wmape_seasonal_naive']:.1%} |
| 28-day moving average | {m['wmape_ma28']:.1%} |
| LightGBM without promo/festival drivers | {m['wmape_lgbm_no_drivers']:.1%} |
| **LightGBM, full** | **{m['wmape_lgbm_full']:.1%}** (bias {m['bias_lgbm_full_pct']:+.1f}%) |

90-day long-range model: WMAPE {m['wmape_long_range_90d']:.1%} vs {m['wmape_same_day_last_year']:.1%} for "same day last year" (90-day hold-out).

Promo-day WMAPE: {m['promo_day_wmape_no_drivers']:.1%} without driver features → {m['promo_day_wmape_full']:.1%} with them.

## Replenishment simulation (28 held-out days, lead time 1 day, ABC service targets)
- **Same stock held:** stockout SKU-days {m['stockout_days_naive']:.2%} → {m['stockout_days_lgbm_same_inventory']:.2%} (**−{m['stockout_reduction_at_same_inventory_pct']:.0f}%**), lost units −{m['lost_units_reduction_at_same_inventory_pct']:.0f}%
- **Same fill rate ({m['fill_rate_lgbm']:.1%}):** overnight stock {m['days_cover_naive_iso']:.2f} → {m['days_cover_lgbm']:.2f} days of cover (**−{m['inventory_reduction_at_same_fill_pct']:.0f}%**)

## Resume bullets (auto-generated from the numbers above)
- Built an end-to-end demand forecasting system for quick-commerce dark stores on **{a['rows'] / 1000:.0f}K+ store-SKU-days**, with DuckDB SQL analytics for ABC–XYZ segmentation, promo uplift, festival/weather effects and stockout-driven lost GMV (₹{a['est_lost_gmv_lakh']:.0f}L).
- Trained a global LightGBM (Tweedie) model with lag, rolling, calendar, promo and festival features and censored-demand correction; **cut 14-day WMAPE by {m['wmape_reduction_vs_naive_pct']:.0f}%** ({m['wmape_seasonal_naive']:.0%} → {m['wmape_lgbm_full']:.0%}) over a seasonal-naive baseline across 3 rolling backtests.
- Converted forecasts into an ABC-tiered order-up-to replenishment policy; simulation showed **{m['stockout_reduction_at_same_inventory_pct']:.0f}% fewer stockouts at equal inventory**, or **{m['inventory_reduction_at_same_fill_pct']:.0f}% less stock at the same {m['fill_rate_lgbm']:.0%} fill rate**.
- Built a short-range (14-day) and long-range (90-day, festival-aware) forecasting setup; the 90-day model beat "same day last year" by {100 * (1 - m['wmape_long_range_90d'] / m['wmape_same_day_last_year']):.0f}% on a 90-day hold-out.
- Shipped an interactive Streamlit app: predict any product-day with promo/festival switches, a promo planner showing unit *and* revenue impact, a sales-history explorer, and a service-level-driven stock plan.
"""
(ROOT / "RESULTS.md").write_text(results, encoding="utf-8")
print(f"\nDone in {time.time() - t0:.0f}s -> RESULTS.md, reports/, models/. Launch: streamlit run app.py")
