"""
Synthetic dark-store sales data that mimics a quick-commerce (Instamart / Blinkit / Zepto)
operation. Used only when no real data is present in data/raw/.

Behaviours baked in (so the analysis has real signal to find):
  - 6 Bengaluru dark stores x 80 SKUs x 18 months of daily sales
  - weekly seasonality that differs by category (snacks/beverages peak on weekends)
  - annual seasonality (summer beverages, monsoon hot snacks, winter dairy)
  - Indian festival spikes (Diwali, Holi, Raksha Bandhan, Christmas/New Year, IPL season)
  - planned promotions with price elasticity
  - rain days raise order volume (people order in instead of going out)
  - business growth trend
  - CENSORED DEMAND: when stock runs out, units_sold < true demand (stockout_flag = 1)
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
OUT = Path(__file__).resolve().parents[1] / "data" / "raw"

STORES = {
    "BLR_KOR": ("Koramangala", 1.25), "BLR_HSR": ("HSR Layout", 1.10),
    "BLR_IND": ("Indiranagar", 1.15), "BLR_WHF": ("Whitefield", 0.95),
    "BLR_JAY": ("Jayanagar", 0.85), "BLR_HEB": ("Hebbal", 0.75),
}

# category: (n_skus, base daily units range, price range, weekend factor, promo elasticity)
CATEGORIES = {
    "Dairy & Bread":   (12, (18, 45), (25, 90), 1.05, 1.2),
    "Fruits & Veg":    (14, (10, 35), (20, 120), 1.10, 1.4),
    "Snacks":          (12, (6, 22), (10, 99), 1.35, 2.2),
    "Beverages":       (10, (5, 20), (20, 150), 1.30, 2.0),
    "Staples":         (10, (4, 12), (60, 450), 0.95, 1.6),
    "Personal Care":   (8, (1.5, 6), (80, 499), 1.00, 1.8),
    "Household":       (7, (1.5, 5), (60, 399), 1.10, 1.7),
    "Frozen & Ice Cream": (7, (2, 9), (60, 350), 1.40, 2.4),
}

FESTIVALS = {  # date: (name, intensity by category key)
    "2024-01-01": "New Year", "2024-01-15": "Sankranti", "2024-03-25": "Holi",
    "2024-04-11": "Eid", "2024-08-15": "Independence Day", "2024-08-19": "Raksha Bandhan",
    "2024-09-07": "Ganesh Chaturthi", "2024-10-12": "Dussehra", "2024-11-01": "Diwali",
    "2024-12-25": "Christmas", "2024-12-31": "New Year's Eve", "2025-01-01": "New Year",
    "2025-01-14": "Sankranti", "2025-03-14": "Holi", "2025-03-31": "Eid",
}
FESTIVAL_LIFT = {"Snacks": 1.8, "Beverages": 1.7, "Frozen & Ice Cream": 1.5, "Dairy & Bread": 1.4,
                 "Staples": 1.3, "Fruits & Veg": 1.2, "Personal Care": 1.1, "Household": 1.25}


def make_calendar(dates):
    cal = pd.DataFrame({"date": dates})
    cal["festival_name"] = cal["date"].dt.strftime("%Y-%m-%d").map(FESTIVALS)
    cal["is_festival"] = cal["festival_name"].notna().astype(int)
    # Diwali build-up: 4 days before Diwali carry a ramp
    diwali = pd.Timestamp("2024-11-01")
    days_to_diwali = (diwali - cal["date"]).dt.days
    cal["diwali_window"] = days_to_diwali.between(0, 4).astype(int)
    # IPL season (evening snacking): late March to late May
    cal["ipl_season"] = (((cal["date"] >= "2024-03-22") & (cal["date"] <= "2024-05-26")) |
                         ((cal["date"] >= "2025-03-22") & (cal["date"] <= "2025-05-25"))).astype(int)
    # Bengaluru rainfall: monsoon Jun-Sep + Oct-Nov retreating monsoon
    m = cal["date"].dt.month
    rain_p = np.select([m.isin([6, 7, 8, 9]), m.isin([10, 11]), m.isin([4, 5])], [0.55, 0.40, 0.2], 0.06)
    rained = RNG.random(len(cal)) < rain_p
    cal["rain_mm"] = np.where(rained, RNG.gamma(2.0, 9.0, len(cal)), 0).round(1)
    return cal


def seasonal_factor(category, doy):
    s = 2 * np.pi * doy / 365.25
    if category in ("Beverages", "Frozen & Ice Cream"):
        return 1 + 0.35 * np.cos(s - 2 * np.pi * 120 / 365.25)  # peak ~ end of April
    if category == "Snacks":
        return 1 + 0.12 * np.cos(s - 2 * np.pi * 210 / 365.25)  # monsoon
    if category == "Fruits & Veg":
        return 1 + 0.10 * np.cos(s - 2 * np.pi * 60 / 365.25)
    return 1 + 0.05 * np.cos(s)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2024-01-01", "2025-06-30", freq="D")
    n = len(dates)
    cal = make_calendar(dates)

    products, sku_i = [], 0
    for cat, (k, base_rng, price_rng, wk, elas) in CATEGORIES.items():
        for j in range(k):
            sku_i += 1
            products.append({
                "sku_id": f"SKU{sku_i:03d}", "sku_name": f"{cat.split(' ')[0]} item {j + 1}",
                "category": cat, "mrp": round(float(RNG.uniform(*price_rng)), 0),
                "base_units": float(np.exp(RNG.uniform(np.log(base_rng[0]), np.log(base_rng[1])))),
                "weekend_factor": wk, "elasticity": elas * RNG.uniform(0.8, 1.2),
                "shelf_life_days": {"Dairy & Bread": 4, "Fruits & Veg": 5}.get(cat, 180),
            })
    prod = pd.DataFrame(products)

    dow = dates.dayofweek.values
    doy = dates.dayofyear.values
    t = np.arange(n) / 365.25
    trend = 1 + 0.28 * t  # 28% YoY growth
    rain_boost = 1 + 0.18 * (cal["rain_mm"].values > 5)

    frames = []
    for store_id, (_, store_mult) in STORES.items():
        store_rand = RNG.uniform(0.9, 1.1, len(prod))
        for i, p in prod.iterrows():
            weekend = np.where(dow >= 5, p.weekend_factor, np.where(dow == 4, 1 + (p.weekend_factor - 1) / 2, 1.0))
            fest = np.where(cal["is_festival"].values == 1, FESTIVAL_LIFT[p.category], 1.0)
            fest = fest * np.where(cal["diwali_window"].values == 1, 1 + (FESTIVAL_LIFT[p.category] - 1) * 0.6, 1.0)
            ipl = np.where(cal["ipl_season"].values == 1, 1.15 if p.category in ("Snacks", "Beverages") else 1.0, 1.0)

            # planned promotions: runs of 3-7 days, ~9% of days
            promo = np.zeros(n, dtype=int)
            disc = np.zeros(n)
            day = int(RNG.integers(0, 20))
            while day < n:
                length = int(RNG.integers(3, 8))
                d = float(RNG.choice([0.10, 0.15, 0.20, 0.25, 0.30]))
                promo[day:day + length] = 1
                disc[day:day + length] = d
                day += length + int(RNG.integers(25, 70))
            promo_lift = 1 + p.elasticity * disc

            mu = (p.base_units * store_mult * store_rand[i] * trend * weekend * seasonal_factor(p.category, doy)
                  * fest * ipl * rain_boost * promo_lift)
            # negative binomial noise (over-dispersed counts)
            r = 12.0
            demand = RNG.negative_binomial(r, r / (r + mu))

            # inventory censoring: a periodic-review replenishment that sometimes falls short
            stock_cap = np.ceil(mu * RNG.uniform(1.35, 2.2, n)).astype(int)
            shortfall_days = RNG.random(n) < (0.025 + 0.04 * (p.category in ("Fruits & Veg", "Dairy & Bread")))
            stock_cap = np.where(shortfall_days, (stock_cap * RNG.uniform(0.3, 0.8, n)).astype(int), stock_cap)
            sold = np.minimum(demand, stock_cap)
            stockout = (demand > stock_cap).astype(int)

            frames.append(pd.DataFrame({
                "date": dates, "store_id": store_id, "sku_id": p.sku_id,
                "units_sold": sold, "selling_price": np.round(p.mrp * (1 - disc), 2),
                "discount_pct": disc, "on_promo": promo, "stockout_flag": stockout,
            }))

    sales = pd.concat(frames, ignore_index=True)
    stores = pd.DataFrame([{"store_id": k, "locality": v[0], "city": "Bengaluru"} for k, v in STORES.items()])
    prod[["sku_id", "sku_name", "category", "mrp", "shelf_life_days"]].to_csv(OUT / "products.csv", index=False)
    stores.to_csv(OUT / "stores.csv", index=False)
    cal.to_csv(OUT / "calendar.csv", index=False)
    sales.to_csv(OUT / "daily_sales.csv", index=False)
    (OUT / ".synthetic").write_text("generated by src/generate_data.py - delete data/raw/* and add real files to replace\n")
    print(f"daily_sales: {len(sales):,} rows | {sales.store_id.nunique()} stores x {sales.sku_id.nunique()} SKUs "
          f"x {sales.date.nunique()} days | stockout rate {sales.stockout_flag.mean():.1%}")


if __name__ == "__main__":
    main()
