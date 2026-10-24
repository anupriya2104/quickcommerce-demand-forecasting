"""Data loading. Real data in data/raw/ always wins; otherwise the synthetic generator runs."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
for p in (REPORTS, FIG):
    p.mkdir(parents=True, exist_ok=True)

REQUIRED = {
    "daily_sales.csv": ["date", "store_id", "sku_id", "units_sold", "selling_price", "discount_pct", "on_promo", "stockout_flag"],
    "products.csv": ["sku_id", "category"],
    "stores.csv": ["store_id", "locality"],
    "calendar.csv": ["date", "festival_name", "is_festival", "rain_mm"],
}


def load_tables():
    if not (RAW / "daily_sales.csv").exists():
        print("No data in data/raw/ -> generating synthetic quick-commerce data")
        from generate_data import main as gen
        gen()
    tables = {}
    for fname, cols in REQUIRED.items():
        df = pd.read_csv(RAW / fname, parse_dates=["date"] if "date" in cols else None)
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"{fname} is missing columns {missing}. See README 'Using real data'.")
        tables[fname.removesuffix(".csv").replace("daily_", "")] = df
    return tables  # keys: sales, products, stores, calendar
