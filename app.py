"""
Streamlit dashboard: business KPIs, forecast explorer and replenishment planner.
Run:  streamlit run app.py      (after `python run_all.py`)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

R = Path(__file__).parent / "reports"
BLUE, GREY, INK = "#2a78d6", "#b9b8b2", "#0b0b0b"
st.set_page_config(page_title="Dark-Store Demand Planner", layout="wide")


@st.cache_data
def load():
    bt = pd.read_parquet(R / "backtest_predictions.parquet")
    return (bt, json.loads((R / "analytics_summary.json").read_text()), json.loads((R / "model_summary.json").read_text()),
            pd.read_csv(R / "sql_abc_xyz.csv"), pd.read_csv(R / "sql_monthly_gmv_by_category.csv", parse_dates=["month"]),
            pd.read_csv(R / "sql_promo_uplift.csv"), pd.read_csv(R / "sql_stockout_by_store.csv"),
            pd.read_csv(R / "model_comparison.csv"))


if not (R / "model_summary.json").exists():
    st.error("Run `python run_all.py` first to build the reports.")
    st.stop()
bt, a, m, abc, gmv, promo, oos, comp = load()

st.title("Dark-Store Demand Forecasting & Replenishment")
st.caption(f"{a['stores']} stores · {a['skus']} SKUs · {a['rows']:,} store-SKU-days · 14-day-ahead global LightGBM")

c = st.columns(5)
c[0].metric("GMV", f"₹{a['gmv_crore']:.1f} Cr")
c[1].metric("Stockout rate", f"{a['stockout_rate_pct']:.1f}% of SKU-days")
c[2].metric("Forecast WMAPE", f"{m['wmape_lgbm_full']:.1%}", f"-{m['wmape_reduction_vs_naive_pct']:.0f}% vs naive", delta_color="inverse")
c[3].metric("Stockouts at same stock", f"{m['stockout_days_lgbm_same_inventory']:.2%}", f"-{m['stockout_reduction_at_same_inventory_pct']:.0f}%", delta_color="inverse")
c[4].metric("Stock at same fill rate", f"{m['days_cover_lgbm']:.2f} days", f"-{m['inventory_reduction_at_same_fill_pct']:.0f}%", delta_color="inverse")

tab1, tab2, tab3, tab4 = st.tabs(["Business overview", "Forecast explorer", "Replenishment planner", "Model report"])

with tab1:
    l, r = st.columns(2)
    fig = px.line(gmv, x="month", y="gmv", color="category", title="Monthly GMV by category (₹)")
    fig.update_layout(legend=dict(orientation="h", y=-0.25), height=420)
    l.plotly_chart(fig, width="stretch")
    fig = px.bar(promo.sort_values("uplift_pct"), x="uplift_pct", y="category", orientation="h",
                 title="Promotion uplift vs same-month baseline (%)", color_discrete_sequence=[BLUE],
                 hover_data=["avg_discount_pct", "promo_events"])
    r.plotly_chart(fig.update_layout(height=420), width="stretch")
    l, r = st.columns(2)
    fig = px.bar(oos.sort_values("est_lost_gmv_lakh"), x="est_lost_gmv_lakh", y="store", orientation="h",
                 title="Estimated GMV lost to stockouts (₹ lakh)", color_discrete_sequence=["#eb6834"],
                 hover_data=["stockout_rate_pct", "est_lost_units"])
    l.plotly_chart(fig.update_layout(height=380), width="stretch")
    seg = abc.groupby(["abc_class", "xyz_class"]).agg(skus=("sku_id", "count"), revenue=("revenue", "sum")).reset_index()
    seg["rev_share_%"] = (100 * seg.revenue / seg.revenue.sum()).round(1)
    r.subheader("ABC–XYZ segmentation")
    r.dataframe(seg.pivot(index="abc_class", columns="xyz_class", values="rev_share_%").fillna(0), width="stretch")
    r.caption("Cell = % of revenue. AX SKUs get the tightest service target (98%); CZ SKUs are candidates for delisting or make-to-order.")

with tab2:
    l, r = st.columns(2)
    store = l.selectbox("Dark store", sorted(bt.store_id.unique()))
    cats = sorted(bt.category.unique())
    cat = r.selectbox("Category", cats)
    skus = sorted(bt[bt.category == cat].sku_id.unique())
    sku = st.selectbox("SKU", skus)
    g = bt[(bt.store_id == store) & (bt.sku_id == sku)].sort_values("date")
    fig = go.Figure()
    for d in g[g.on_promo == 1].date:
        fig.add_vrect(x0=d - pd.Timedelta(hours=12), x1=d + pd.Timedelta(hours=12), fillcolor="#eda100", opacity=0.15, line_width=0)
    fig.add_scatter(x=g.date, y=g.demand, name="Actual demand", line=dict(color=INK))
    fig.add_scatter(x=g.date, y=g.p_seasonal_naive, name="Seasonal naive", line=dict(color=GREY))
    fig.add_scatter(x=g.date, y=g.p_lgbm_full, name="LightGBM", line=dict(color=BLUE, width=3))
    fig.update_layout(title=f"{sku} @ {store} — 3 rolling 14-day backtests (shaded = promo)", hovermode="x unified", height=440)
    st.plotly_chart(fig, width="stretch")
    e = g[g.stockout_flag == 0]
    k = st.columns(3)
    k[0].metric("WMAPE (LightGBM)", f"{np.abs(e.demand - e.p_lgbm_full).sum() / e.demand.sum():.1%}")
    k[1].metric("WMAPE (naive)", f"{np.abs(e.demand - e.p_seasonal_naive).sum() / e.demand.sum():.1%}")
    k[2].metric("ABC / XYZ", "/".join(abc.loc[abc.sku_id == sku, ["abc_class", "xyz_class"]].iloc[0]))

with tab3:
    st.markdown("Order-up-to policy: **target stock = tomorrow's forecast + z × σ(forecast error)**, z set by ABC class.")
    l, m_, r = st.columns(3)
    zA = l.slider("Service level, A SKUs", 0.90, 0.995, 0.98)
    zB = m_.slider("Service level, B SKUs", 0.85, 0.99, 0.95)
    zC = r.slider("Service level, C SKUs", 0.80, 0.98, 0.90)
    from scipy.stats import norm
    zmap = {"A": norm.ppf(zA), "B": norm.ppf(zB), "C": norm.ppf(zC)}
    store_p = st.selectbox("Plan for store", sorted(bt.store_id.unique()), key="plan")
    cal = bt[bt.fold == 1]
    sig = (cal.demand - cal.p_lgbm_full).groupby([cal.store_id, cal.sku_id]).std().rename("sigma").reset_index()
    day = bt.date.max()
    plan = (bt[(bt.date == day) & (bt.store_id == store_p)].merge(sig, on=["store_id", "sku_id"])
            .merge(abc[["sku_id", "abc_class", "xyz_class"]], on="sku_id"))
    plan["forecast"] = plan.p_lgbm_full.round(1)
    plan["safety_stock"] = np.ceil(plan.abc_class.map(zmap) * plan.sigma)
    plan["order_up_to"] = np.ceil(plan.forecast + plan.safety_stock).astype(int)
    st.caption(f"Plan for {day.date()} · {len(plan)} SKUs · total target stock {plan.order_up_to.sum():,} units")
    st.dataframe(plan[["sku_id", "category", "abc_class", "xyz_class", "forecast", "safety_stock", "order_up_to"]]
                 .sort_values(["abc_class", "order_up_to"], ascending=[True, False]), width="stretch", height=420)

with tab4:
    fig = px.bar(comp.sort_values("wmape", ascending=False), x="wmape", y="model", orientation="h",
                 title="WMAPE across 3 rolling-origin backtests (lower is better)", color_discrete_sequence=[BLUE])
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig.update_layout(height=360), width="stretch")
    st.dataframe(comp.round(4), width="stretch")
    for f in ("09_feature_importance.png", "10_inventory_simulation.png"):
        st.image(str(R / "figures" / f))
