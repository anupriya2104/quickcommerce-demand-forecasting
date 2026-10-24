"""
Convert the real Kaggle M5 Forecasting dataset (Walmart) into this project's schema.

Download from https://www.kaggle.com/competitions/m5-forecasting-accuracy/data :
  sales_train_evaluation.csv, calendar.csv, sell_prices.csv  ->  put them in data/m5/

Then:  python src/adapt_m5.py --stores CA_1 CA_2 CA_3 TX_1 TX_2 WI_1 --top-skus 80
       python run_all.py

Mapping:
  store_id      <- M5 store_id               sku_id        <- M5 item_id
  category      <- M5 dept_id                units_sold    <- daily sales
  selling_price <- weekly sell_price         on_promo      <- price >= 7% below the item-store's rolling max
  stockout_flag <- zero sales for 7+ days on an item that normally sells (inferred, M5 has no stock data)
  festival_name <- M5 event_name_1           rain_mm       <- not available (0)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "data" / "m5", ROOT / "data" / "raw"


def main(stores, top_skus, start):
    cal = pd.read_csv(SRC / "calendar.csv", parse_dates=["date"])
    wide = pd.read_csv(SRC / "sales_train_evaluation.csv")
    wide = wide[wide.store_id.isin(stores)]
    day_cols = [c for c in wide.columns if c.startswith("d_")]
    # keep the highest-volume items so the run stays laptop-sized
    top = wide.groupby("item_id")[day_cols].sum().sum(axis=1).nlargest(top_skus).index
    wide = wide[wide.item_id.isin(top)]
    long = wide.melt(id_vars=["item_id", "dept_id", "store_id"], value_vars=day_cols, var_name="d", value_name="units_sold")
    long = long.merge(cal[["d", "date", "wm_yr_wk", "event_name_1"]], on="d")
    long = long[long.date >= start]

    prices = pd.read_csv(SRC / "sell_prices.csv")
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left").sort_values(["store_id", "item_id", "date"])
    long = long.dropna(subset=["sell_price"])  # item not yet listed in that store
    g = long.groupby(["store_id", "item_id"])
    ref = g.sell_price.transform(lambda x: x.rolling(26 * 7, min_periods=1).max())
    long["discount_pct"] = (1 - long.sell_price / ref).clip(lower=0).round(3)
    long["on_promo"] = (long.discount_pct >= 0.07).astype(int)

    zero = (long.units_sold == 0).astype(int)
    run = zero.groupby([long.store_id, long.item_id, (zero == 0).cumsum()]).cumsum()
    normal_rate = g.units_sold.transform(lambda x: x.rolling(56, min_periods=14).mean().shift(1))
    long["stockout_flag"] = ((run >= 7) & (normal_rate > 1)).astype(int)

    sales = long.rename(columns={"item_id": "sku_id", "sell_price": "selling_price"})[
        ["date", "store_id", "sku_id", "units_sold", "selling_price", "discount_pct", "on_promo", "stockout_flag"]]
    products = long.drop_duplicates("item_id")[["item_id", "dept_id"]].rename(columns={"item_id": "sku_id", "dept_id": "category"})
    stores_df = pd.DataFrame({"store_id": stores, "locality": stores})
    calendar = cal[cal.date >= start][["date", "event_name_1"]].rename(columns={"event_name_1": "festival_name"})
    calendar["is_festival"] = calendar.festival_name.notna().astype(int)
    calendar["diwali_window"] = 0  # placeholders kept so the feature set stays identical
    calendar["ipl_season"] = 0
    calendar["rain_mm"] = 0.0

    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*"):
        f.unlink()
    sales.to_csv(OUT / "daily_sales.csv", index=False)
    products.to_csv(OUT / "products.csv", index=False)
    stores_df.to_csv(OUT / "stores.csv", index=False)
    calendar.to_csv(OUT / "calendar.csv", index=False)
    print(f"Wrote {len(sales):,} rows: {sales.store_id.nunique()} stores x {sales.sku_id.nunique()} items, "
          f"{sales.date.min().date()} -> {sales.date.max().date()}, promo days {sales.on_promo.mean():.1%}, "
          f"inferred stockouts {sales.stockout_flag.mean():.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", nargs="+", default=["CA_1", "CA_2", "CA_3", "TX_1", "TX_2", "WI_1"])
    ap.add_argument("--top-skus", type=int, default=80)
    ap.add_argument("--start", default="2014-07-01")
    a = ap.parse_args()
    main(a.stores, a.top_skus, a.start)
