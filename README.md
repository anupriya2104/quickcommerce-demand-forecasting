# Quick-Commerce Demand Forecasting & Inventory Optimization

**Problem.** A 10-minute-delivery dark store holds ~1 day of stock. Over-order and perishables
expire; under-order and the customer sees "out of stock" and switches apps. The store needs a
14-day demand forecast per SKU and a rule that turns it into tomorrow's order.

**What this project does** — analytics and data science in one pipeline:

| Stage | Analytics side | Data-science side |
|---|---|---|
| 1. SQL analytics (`sql/analytics.sql`, DuckDB) | GMV trends, YoY growth, ABC–XYZ segmentation, day-of-week index, promo uplift, festival & rain effects, lost GMV from stockouts | — |
| 2. Forecasting (`src/forecast.py`) | Error by category and on promo days | Global LightGBM (Tweedie) over all store-SKUs, 3 rolling-origin backtests, 4 baselines, ablation of driver features, censored-demand correction |
| 3. Inventory simulation | Fill rate, stockout days, days of cover | ABC-tiered order-up-to policy using forecast-error σ, iso-service and iso-inventory comparisons |
| 4. Dashboard (`app.py`) | KPI tiles, GMV/promo/stockout views | SKU forecast drill-down, live replenishment planner with service-level sliders |

See **[RESULTS.md](RESULTS.md)** for the numbers (regenerated on every run).

## Run it

```bash
pip install -r requirements.txt
python run_all.py          # ~1 min: generates data if none, runs SQL + models + simulation, writes RESULTS.md
streamlit run app.py       # dashboard at http://localhost:8501
```

## Using real data (do this before quoting numbers)

The default run uses `src/generate_data.py`, a simulator of 6 Bengaluru dark stores × 80 SKUs × 18 months
with realistic weekly/annual seasonality, Indian festivals, IPL season, monsoon, promotions with price
elasticity, and **censored demand** (sales capped by stock). Real quick-commerce data is not public, so for
real numbers use the **M5 (Walmart) dataset**:

1. Download `sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv` from
   [Kaggle M5](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) into `data/m5/`
2. `python src/adapt_m5.py --stores CA_1 CA_2 CA_3 TX_1 TX_2 WI_1 --top-skus 80`
3. `python run_all.py`

The adapter infers promotions from price drops and stockouts from abnormal zero-sales runs. Currency
labels in charts say ₹; change them if you present the M5 version.

## Key design decisions (interview talking points)

- **Why a 14-day horizon with lags ≥ 14?** Every feature must be known when the plan is made.
  Using `lag_1` would score beautifully in a random split and be useless in production.
- **Why rolling-origin backtests, not a random split?** Random splits leak the future into training
  for time series. Three successive 14-day windows also show the error is stable over time.
- **Censored demand.** On a stockout day, sales = stock, not demand. Training on that teaches the model to
  under-forecast exactly the items that sell out. Fix: drop stockout days from the target and impute them
  for lag features.
- **One global model vs 480 local models.** A single LightGBM learns shared patterns (festival lift,
  weekend effect) across SKUs, which helps low-volume items, and trains in seconds.
- **Tweedie objective.** Daily SKU demand is non-negative, skewed and has zeros; Tweedie fits that better than MSE.
- **WMAPE, not MAPE.** MAPE explodes on low-volume days; WMAPE weights errors by volume, which is what costs money.
- **The ablation.** Removing promo/festival features costs ~1 WMAPE point overall but ~5 points on promo days —
  the drivers matter exactly when the business is spending on discounts.
- **Fair inventory comparison.** A better forecast can be cashed in two ways: fewer stockouts at the same stock, or
  less stock at the same service. Both are reported; comparing at equal safety factors would be misleading.
- **ABC–XYZ.** Service targets differ by class: A items (80% of revenue) get 98%, C items 90%.

## Project layout

```
├── run_all.py                 # one-command pipeline, writes RESULTS.md
├── app.py                     # Streamlit dashboard
├── sql/analytics.sql          # 9 named business queries (CTEs, window functions, FILTER)
├── src/
│   ├── generate_data.py       # dark-store simulator (used when data/raw is empty)
│   ├── adapt_m5.py            # real-data adapter for Kaggle M5
│   ├── data.py                # loading + schema checks
│   ├── analytics.py           # runs the SQL, builds charts
│   ├── forecast.py            # features, backtests, models, inventory simulation
│   └── viz.py                 # chart style
├── reports/                   # query outputs, metrics JSON, backtest predictions, figures/
└── models/                    # trained LightGBM model
```
