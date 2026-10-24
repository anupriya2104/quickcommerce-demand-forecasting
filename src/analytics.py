"""Stage 1 - business analytics: runs sql/analytics.sql in DuckDB, saves tables + charts."""
import json
import re

import duckdb
import numpy as np
import pandas as pd

from data import FIG, REPORTS, ROOT, load_tables
from viz import MUTED, SEQ_BLUE, SERIES, plt, save

DOW = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]  # DuckDB DAYOFWEEK: 0 = Sunday


def run_sql(con):
    text = (ROOT / "sql" / "analytics.sql").read_text()
    blocks = re.split(r"--\s*name:\s*(\w+)", text)[1:]
    results = {}
    for name, sql in zip(blocks[::2], blocks[1::2]):
        results[name] = con.execute(sql).df()
        results[name].to_csv(REPORTS / f"sql_{name}.csv", index=False)
    return results


def charts(r):
    # 1. Monthly GMV by category (stacked would hide trends -> small lines, top 4 + other)
    m = r["monthly_gmv_by_category"].copy()
    top = m.groupby("category").gmv.sum().nlargest(4).index
    m["cat"] = np.where(m.category.isin(top), m.category, "Other")
    pv = m.groupby(["month", "cat"]).gmv.sum().unstack() / 1e5
    pv = pv[list(top) + (["Other"] if "Other" in pv.columns else [])]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, c in enumerate(pv.columns):
        ax.plot(pv.index, pv[c], color=SERIES[i] if c != "Other" else MUTED, label=c, marker="o", ms=3)
    ax.set_title("Monthly GMV by category (₹ lakh)")
    ax.set_ylabel("GMV, ₹ lakh")
    ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(0, -0.12))
    save(fig, FIG / "01_monthly_gmv_by_category.png")

    # 2. ABC-XYZ matrix
    ax_ = r["abc_xyz"]
    mat = pd.crosstab(ax_.abc_class, ax_.xyz_class).reindex(index=list("ABC"), columns=list("XYZ"), fill_value=0)
    rev = ax_.groupby(["abc_class", "xyz_class"]).revenue.sum().unstack().reindex(index=list("ABC"), columns=list("XYZ")).fillna(0)
    rev_share = rev / rev.values.sum() * 100
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.imshow(rev_share.values, cmap=plt.matplotlib.colors.ListedColormap(SEQ_BLUE[:5]), vmin=0, vmax=rev_share.values.max())
    for i in range(3):
        for j in range(3):
            v = rev_share.values[i, j]
            ax.text(j, i, f"{mat.values[i, j]} SKUs\n{v:.0f}% rev", ha="center", va="center",
                    color="white" if v > rev_share.values.max() * 0.55 else "#0b0b0b", fontsize=9)
    ax.set_xticks(range(3), ["X (stable)", "Y (variable)", "Z (erratic)"])
    ax.set_yticks(range(3), ["A (top 80% rev)", "B (next 15%)", "C (last 5%)"])
    ax.grid(False)
    ax.set_title("ABC–XYZ inventory segmentation")
    save(fig, FIG / "02_abc_xyz_matrix.png")

    # 3. Day-of-week index heatmap
    d = r["dow_index"].pivot(index="category", columns="dow", values="demand_index")
    d = d[[1, 2, 3, 4, 5, 6, 0]]
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(d.values, cmap="RdBu_r", vmin=70, vmax=130, aspect="auto")
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            ax.text(j, i, f"{d.values[i, j]:.0f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_yticks(range(len(d)), d.index)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Index (100 = avg day)", shrink=0.8)
    ax.set_title("Day-of-week demand index (non-promo, in-stock days)")
    save(fig, FIG / "03_dow_index.png")

    # 4. Promo uplift by category
    p = r["promo_uplift"].sort_values("uplift_pct")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(p.category, p.uplift_pct, color=SERIES[0], height=0.6)
    for y, (v, dsc) in enumerate(zip(p.uplift_pct, p.avg_discount_pct)):
        ax.text(v + 0.5, y, f"+{v:.0f}%  (avg {dsc:.0f}% off)", va="center", fontsize=8, color=MUTED)
    ax.set_xlabel("Unit uplift vs same-month baseline, %")
    ax.set_title("Promotion uplift by category")
    ax.set_xlim(0, p.uplift_pct.max() * 1.45)
    save(fig, FIG / "04_promo_uplift.png")

    # 5. Lost GMV from stockouts by store
    s = r["stockout_by_store"].sort_values("est_lost_gmv_lakh")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(s.store, s.est_lost_gmv_lakh, color=SERIES[1], height=0.6)
    for y, (v, rt) in enumerate(zip(s.est_lost_gmv_lakh, s.stockout_rate_pct)):
        ax.text(v, y, f"  ₹{v:.1f}L  ({rt:.1f}% SKU-days OOS)", va="center", fontsize=8, color=MUTED)
    ax.set_xlim(0, s.est_lost_gmv_lakh.max() * 1.5)
    ax.set_xlabel("Estimated lost GMV, ₹ lakh")
    ax.set_title("Revenue lost to stockouts, by dark store")
    save(fig, FIG / "05_stockout_lost_gmv.png")


def main():
    t = load_tables()
    con = duckdb.connect()
    for k, v in t.items():
        con.register(k, v)
    r = run_sql(con)
    charts(r)

    k = r["kpi_overview"].iloc[0]
    rain = r["rain_effect"].set_index("weather").units_per_day
    abc = r["abc_xyz"]
    insights = {
        "rows": int(len(t["sales"])), "stores": int(k.stores), "skus": int(k.skus),
        "gmv_crore": float(k.gmv_crore), "stockout_rate_pct": float(k.stockout_rate_pct),
        "a_class_skus": int((abc.abc_class == "A").sum()),
        "a_class_sku_share_pct": round(100 * (abc.abc_class == "A").mean(), 1),
        "top_promo_category": r["promo_uplift"].iloc[0].category,
        "top_promo_uplift_pct": float(r["promo_uplift"].iloc[0].uplift_pct),
        "rain_day_lift_pct": round(100 * (rain.get("Rain day (>5mm)", np.nan) / rain.get("Dry day", np.nan) - 1), 1),
        "est_lost_gmv_lakh": float(r["stockout_by_store"].est_lost_gmv_lakh.sum()),
        "top_yoy_category": r["category_yoy_growth"].iloc[0].category,
        "top_yoy_pct": float(r["category_yoy_growth"].iloc[0].yoy_growth_pct),
    }
    (REPORTS / "analytics_summary.json").write_text(json.dumps(insights, indent=2))
    print(json.dumps(insights, indent=2))


if __name__ == "__main__":
    main()
