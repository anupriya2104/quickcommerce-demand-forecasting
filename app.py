"""
QuickCart Demand Studio - interactive demand forecasting dashboard for quick-commerce dark stores.

Tabs: Predict a day · Next 90 days · Promo planner · Sales history · Order plan · Business insights · Model report
Run:  streamlit run app.py      (after `python run_all.py`)
"""
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from scipy.stats import norm

ROOT = Path(__file__).parent
R = ROOT / "reports"
st.set_page_config(page_title="QuickCart Demand Studio", page_icon="🛒", layout="wide")

# ----------------------------------------------------------------------------------- look & feel
ORANGE, MAGENTA, PURPLE, GREEN, YELLOW, BLUE = "#F46A1F", "#D6246E", "#6C3CE1", "#0E9F6E", "#FFB020", "#2874F0"
INK, MUTED, LIGHT, GRID = "#2B2B3A", "#6B6B80", "#C9C6CF", "#F3ECE6"
EMOJI = {"Dairy & Bread": "🥛", "Fruits & Veg": "🥦", "Snacks": "🍿", "Beverages": "🥤", "Staples": "🍚",
         "Personal Care": "🧴", "Household": "🧽", "Frozen & Ice Cream": "🍦"}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

pio.templates["quickcart"] = go.layout.Template(layout=dict(
    font=dict(family="Poppins, sans-serif", color=INK, size=13), paper_bgcolor="white", plot_bgcolor="white",
    colorway=[ORANGE, PURPLE, GREEN, MAGENTA, BLUE, YELLOW], hoverlabel=dict(font_family="Poppins, sans-serif"),
    xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID), yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID),
    margin=dict(l=10, r=10, t=50, b=10), title=dict(font=dict(size=16, color=INK), x=0.01),
    legend=dict(orientation="h", y=-0.18, x=0)))
pio.templates.default = "quickcart"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, select, textarea {{ font-family: 'Poppins', sans-serif; }}
.block-container {{ padding-top: 1.2rem; max-width: 1350px; }}
.hero {{ background: linear-gradient(115deg, {ORANGE} 0%, {MAGENTA} 55%, {PURPLE} 100%); border-radius: 22px;
        padding: 22px 28px; color: white; margin-bottom: 14px; box-shadow: 0 10px 30px rgba(214,36,110,.18); }}
.hero h1 {{ color: white; font-size: 2.0rem; margin: 0; font-weight: 700; padding: 0; }}
.hero p {{ margin: 4px 0 10px 0; opacity: .95; color: white; }}
.chip {{ display:inline-block; background: rgba(255,255,255,.2); border: 1px solid rgba(255,255,255,.35);
        padding: 3px 12px; border-radius: 999px; margin: 2px 6px 2px 0; font-size: .82rem; color: white; }}
[data-testid="stMetric"] {{ background: white; border: 1px solid #F1E4D8; border-radius: 16px; padding: 12px 16px;
        box-shadow: 0 3px 12px rgba(244,106,31,.07); }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-weight: 500; }}
[data-testid="stMetricValue"] {{ color: {INK}; font-weight: 700; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; flex-wrap: wrap; }}
.stTabs [data-baseweb="tab"] {{ background: white; border: 1px solid #F1E4D8; border-radius: 999px; padding: 6px 16px; height: auto; }}
.stTabs [aria-selected="true"] {{ background: {ORANGE}; border-color: {ORANGE}; }}
.stTabs [aria-selected="true"] p {{ color: white !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}
.card {{ background: white; border: 1px solid #F1E4D8; border-radius: 18px; padding: 18px 20px;
        box-shadow: 0 4px 14px rgba(43,43,58,.05); }}
.big-number {{ font-size: 3rem; font-weight: 700; line-height: 1; color: {ORANGE}; }}
.unit {{ font-size: 1.1rem; color: {MUTED}; font-weight: 500; }}
.badge {{ display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600; margin: 2px 4px 2px 0; }}
.b-green {{ background:#E3F6EE; color:{GREEN}; }} .b-purple {{ background:#EEE8FD; color:{PURPLE}; }}
.b-orange {{ background:#FFEBDD; color:{ORANGE}; }} .b-pink {{ background:#FCE4EE; color:{MAGENTA}; }}
.b-yellow {{ background:#FFF4D6; color:#9A6A00; }}
.product-card {{ background: white; border-radius: 18px; padding: 14px; border: 1px solid #F1E4D8; text-align: center; }}
.product-card .emoji {{ font-size: 2.6rem; }}
.product-card .name {{ font-weight: 600; font-size: 1.02rem; margin-top: 2px; }}
.muted {{ color: {MUTED}; font-size: .85rem; }}
section[data-testid="stSidebar"] {{ background: #FFF1E6; }}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------------- data
@st.cache_resource
def load_models():
    meta = joblib.load(ROOT / "models" / "meta.joblib")
    return (lgb.Booster(model_file=str(ROOT / "models" / "lgbm_demand.txt")),
            lgb.Booster(model_file=str(ROOT / "models" / "lgbm_demand_long.txt")), meta)


@st.cache_data
def load_data():
    def j(f):
        return json.loads((R / f).read_text(encoding="utf-8"))
    names = pd.read_csv(R / "products.csv") if (R / "products.csv").exists() else None
    stores = pd.read_csv(R / "stores.csv") if (R / "stores.csv").exists() else None
    return (pd.read_parquet(R / "future_features.parquet"), pd.read_parquet(R / "sales_history.parquet"),
            pd.read_parquet(R / "backtest_predictions.parquet"), j("analytics_summary.json"), j("model_summary.json"),
            pd.read_csv(R / "sql_abc_xyz.csv"), pd.read_csv(R / "sql_monthly_gmv_by_category.csv", parse_dates=["month"]),
            pd.read_csv(R / "sql_promo_uplift.csv"), pd.read_csv(R / "sql_stockout_by_store.csv"),
            pd.read_csv(R / "model_comparison.csv"), names, stores)


if not (R / "future_features.parquet").exists() or not (ROOT / "models" / "lgbm_demand_long.txt").exists():
    st.error("Run `python run_all.py` first to build the forecasts.")
    st.stop()

short_m, long_m, META = load_models()
SHORT_F, LONG_F, H = META["features"], META["long_features"], META["horizon"]
fut, hist, bt, A, M, abc, gmv, promo, oos, comp, names, stores = load_data()
name_of = dict(zip(names.sku_id, names.sku_name)) if names is not None else {}
store_name = dict(zip(stores.store_id, stores.locality)) if stores is not None else {}


def emoji(c):
    return EMOJI.get(c, "📦")


def predict(df):
    """Short-range model for days 1-14, long-range model beyond."""
    out = np.empty(len(df))
    s_mask = (df.horizon_day <= H).values
    if s_mask.any():
        out[s_mask] = short_m.predict(df.loc[s_mask, SHORT_F])
    if (~s_mask).any():
        out[~s_mask] = long_m.predict(df.loc[~s_mask, LONG_F])
    return np.clip(out, 0, None)


def with_conditions(df, promo_mask=None, disc=0.0, festival=None, known_fest=False):
    """Copy of the future rows with a promo plan and (optionally) a festival switch applied."""
    df = df.copy()
    df["selling_price"] = df.regular_price
    df["on_promo"], df["discount_pct"] = 0, 0.0
    if promo_mask is not None and np.any(promo_mask):
        df.loc[promo_mask, "on_promo"] = 1
        df.loc[promo_mask, "discount_pct"] = disc
        df.loc[promo_mask, "selling_price"] = df.loc[promo_mask, "regular_price"] * (1 - disc)
    df["price_index"] = df.selling_price / df.max_price
    if festival is True:
        df["is_festival"], df["days_to_festival"], df["days_since_festival"] = 1, 0, 0
    elif festival is False:
        df["is_festival"] = 0
        if known_fest:  # a real festival switched off -> treat as an ordinary day
            df["days_to_festival"], df["days_since_festival"] = 30, 30
    return df


def badge(text, kind):
    return f'<span class="badge b-{kind}">{text}</span>'


# ----------------------------------------------------------------------------------- sidebar: pick a product
with st.sidebar:
    st.markdown("### 🛒 Pick a product")
    store = st.selectbox("Dark store", sorted(fut.store_id.unique()), format_func=lambda s: f"📍 {store_name.get(s, s)}")
    cat = st.selectbox("Category", sorted(fut.category.unique()), format_func=lambda c: f"{emoji(c)} {c}")
    skus = sorted(fut[fut.category == cat].sku_id.unique(), key=lambda s: name_of.get(s, s))
    sku = st.selectbox("Product", skus, format_func=lambda s: name_of.get(s, s))
    pf = fut[(fut.store_id == store) & (fut.sku_id == sku)].sort_values("date").reset_index(drop=True)
    ph = hist[(hist.store_id == store) & (hist.sku_id == sku)].sort_values("date").reset_index(drop=True)
    cls = abc.loc[abc.sku_id == sku, ["abc_class", "xyz_class"]].iloc[0]
    steadiness = {"X": "steady", "Y": "variable", "Z": "erratic"}[cls.xyz_class]
    st.markdown(f"""<div class="product-card"><div class="emoji">{emoji(cat)}</div>
        <div class="name">{name_of.get(sku, sku)}</div><div class="muted">{sku} · {store_name.get(store, store)}</div>
        <div style="margin-top:8px">{badge('Class ' + cls.abc_class, 'orange')}{badge('Demand ' + steadiness, 'purple')}</div>
        <div style="margin-top:8px"><b>₹{pf.regular_price.iloc[0]:,.0f}</b> <span class="muted">regular price</span></div>
        <div><b>{ph.tail(28).demand.mean():.0f}</b> <span class="muted">units/day, last 4 weeks</span></div></div>""",
                unsafe_allow_html=True)
    st.caption("All tabs follow this selection. Data: simulated dark-store sales (see README).")

# ----------------------------------------------------------------------------------- hero + KPIs
first, last_f = fut.date.min(), fut.date.max()
st.markdown(f"""<div class="hero"><h1>🛒 QuickCart Demand Studio</h1>
<p>Predict what will sell, when, and how much to stock — for every product in every dark store.</p>
<span class="chip">📍 {A['stores']} dark stores</span><span class="chip">🧺 {A['skus']} products</span>
<span class="chip">📜 history {hist.date.min():%b %Y} – {hist.date.max():%b %Y}</span>
<span class="chip">🔮 forecasts to {last_f:%d %b %Y}</span></div>""", unsafe_allow_html=True)
k = st.columns(5)
k[0].metric("Sales value (GMV)", f"₹{A['gmv_crore']:.1f} Cr")
k[1].metric("Forecast error (WMAPE)", f"{M['wmape_lgbm_full']:.0%}", f"{M['wmape_reduction_vs_naive_pct']:.0f}% better than naive")
k[2].metric("Stockouts at same stock", f"−{M['stockout_reduction_at_same_inventory_pct']:.0f}%", "vs naive planning", delta_color="off")
k[3].metric("Stock needed, same service", f"−{M['inventory_reduction_at_same_fill_pct']:.0f}%", "vs naive planning", delta_color="off")
k[4].metric("Revenue lost to stockouts", f"₹{A['est_lost_gmv_lakh']:.0f} L", "across all stores", delta_color="off")

tabs = st.tabs(["🔮 Predict a day", "📅 Next 90 days", "🏷️ Promo planner", "📜 Sales history", "📦 Order plan",
                "📊 Business insights", "🧠 Model report"])

# =================================================================================== 1. PREDICT A DAY
with tabs[0]:
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.markdown(f"#### {emoji(cat)} {name_of.get(sku, sku)}")
        day = pd.Timestamp(st.date_input("Which day?", value=first.date(), min_value=first.date(), max_value=last_f.date(),
                                          help=f"Any day from {first:%d %b} to {last_f:%d %b %Y}", key="p_day"))
        row = pf[pf.date == day]
        known_fest = row.festival_name.iloc[0] if row.festival_name.notna().iloc[0] else None
        info = badge(f"📆 {day:%A}", "purple") + (badge("🎈 Weekend", "pink") if day.dayofweek >= 5 else "")
        info += badge(f"🎉 {known_fest}", "yellow") if known_fest else ""
        st.markdown(info, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        promo_on = c1.toggle("🏷️ Run a discount", key="p_promo")
        fest_on = c2.toggle("🎉 Festival day", value=bool(known_fest), key=f"p_fest_{day:%Y%m%d}")
        disc = st.slider("Discount %", 5, 50, 20, 5, disabled=not promo_on, key="p_disc")
        sl = st.slider("How sure do you want to be of not running out?", 0.80, 0.995, 0.95, format="%.3f",
                       help="Service level. Higher = more safety stock.", key="p_sl")

    one = np.array([True])
    x = with_conditions(row, one if promo_on else None, disc / 100, festival=fest_on, known_fest=bool(known_fest))
    pred = float(predict(x)[0])
    base = float(predict(with_conditions(row, None, 0, festival=False, known_fest=bool(known_fest)))[0])
    sigma = float(row.sigma.fillna(pf.sigma.mean()).iloc[0])
    lo, hi = max(0.0, pred - 1.28 * sigma), pred + 1.28 * sigma
    stock = pred + norm.ppf(sl) * sigma
    price = float(x.selling_price.iloc[0])
    h = int(row.horizon_day.iloc[0])
    model_badge = (badge("⚡ Short-range model · high confidence", "green") if h <= H
                   else badge("🗓️ Long-range model · planning estimate", "purple"))
    diff = pred - base
    diff_txt = (f"<span style='color:{GREEN};font-weight:600'>▲ {diff:.0f} units vs a normal day</span>" if diff > 0.5 else
                f"<span style='color:{MAGENTA};font-weight:600'>▼ {abs(diff):.0f} units vs a normal day</span>" if diff < -0.5 else
                "<span class='muted'>same as a normal day</span>")
    same_dow = ph[ph.date.dt.dayofweek == day.dayofweek].tail(8).demand
    far_note = "" if h <= H else " More than 14 days ahead, recent weekly sales can't be used yet, so the range is wider."
    with right:
        st.markdown(f"""<div class="card">{model_badge}
            <div class="muted" style="margin-top:10px">Predicted demand on {day:%a %d %b %Y}</div>
            <div><span class="big-number">{pred:.0f}</span> <span class="unit">units</span></div>
            <div style="margin:4px 0 12px 0">{diff_txt}</div>
            <div style="display:flex; gap:28px; flex-wrap:wrap">
              <div><div class="muted">Likely range (80%)</div><b style="font-size:1.25rem">{lo:.0f} – {hi:.0f}</b></div>
              <div><div class="muted">Stock to keep</div><b style="font-size:1.25rem;color:{PURPLE}">{np.ceil(stock):.0f} units</b></div>
              <div><div class="muted">Expected revenue</div><b style="font-size:1.25rem;color:{GREEN}">₹{pred * price:,.0f}</b></div>
            </div>
            <div class="muted" style="margin-top:12px">Last 8 {day:%A}s: {same_dow.min():.0f}–{same_dow.max():.0f} units
            (avg {same_dow.mean():.0f}).{far_note}</div></div>""", unsafe_allow_html=True)

    st.markdown("##### What changes the prediction")
    scen = []
    for label, p, f in [("Normal day", False, False), (f"With {disc}% off", True, False), ("Festival day", False, True),
                        (f"Festival + {disc}% off", True, True)]:
        z = with_conditions(row, one if p else None, disc / 100, festival=f, known_fest=bool(known_fest))
        scen.append((label, float(predict(z)[0])))
    sc = pd.DataFrame(scen, columns=["Scenario", "Units"])
    fig = go.Figure(go.Bar(x=sc.Scenario, y=sc.Units, marker_color=[LIGHT, ORANGE, YELLOW, MAGENTA],
                           text=sc.Units.round(0).astype(int), textposition="outside"))
    fig.update_layout(height=300, yaxis_title="Predicted units", title=f"{name_of.get(sku, sku)} on {day:%a %d %b}",
                      yaxis_range=[0, sc.Units.max() * 1.25])
    st.plotly_chart(fig, width="stretch")

# =================================================================================== 2. NEXT 90 DAYS
with tabs[1]:
    st.markdown(f"#### {emoji(cat)} {name_of.get(sku, sku)} — forecast for the next 90 days")
    band = 1.28 * pf.sigma.fillna(pf.sigma.mean())
    recent_hist = ph[ph.date > ph.date.max() - pd.Timedelta(days=42)]
    fig = go.Figure()
    fig.add_vrect(x0=first - pd.Timedelta(hours=12), x1=first + pd.Timedelta(days=H - 0.5), fillcolor=GREEN, opacity=0.07,
                  line_width=0, annotation_text="next 14 days: high confidence", annotation_position="top left",
                  annotation_font_color=GREEN)
    fig.add_scatter(x=pd.concat([pf.date, pf.date[::-1]]), y=pd.concat([pf.forecast + band, (pf.forecast - band).clip(lower=0)[::-1]]),
                    fill="toself", fillcolor="rgba(244,106,31,0.13)", line=dict(width=0), name="likely range (80%)", hoverinfo="skip")
    fig.add_scatter(x=recent_hist.date, y=recent_hist.demand, name="actual sales", line=dict(color=INK, width=1.6))
    fig.add_scatter(x=pf.date, y=pf.forecast, name="forecast", line=dict(color=ORANGE, width=3),
                    hovertemplate="%{x|%a %d %b}: %{y:.0f} units<extra></extra>")
    wk = pf[pf.date.dt.dayofweek >= 5]
    fig.add_scatter(x=wk.date, y=wk.forecast, mode="markers", name="weekend", marker=dict(color=PURPLE, size=7))
    for _, fr in pf[pf.festival_name.notna()].iterrows():
        fig.add_vline(x=fr.date, line_dash="dot", line_color=MAGENTA)
        fig.add_annotation(x=fr.date, y=1.02, yref="paper", text=f"🎉 {fr.festival_name}", showarrow=False,
                           font=dict(color=MAGENTA, size=11), textangle=-30, xanchor="left")
    fig.update_layout(height=460, hovermode="x unified", yaxis_title="Units / day", margin=dict(t=80))
    st.plotly_chart(fig, width="stretch")

    c = st.columns(4)
    c[0].metric("Next 7 days", f"{pf.forecast.head(7).sum():,.0f} units")
    c[1].metric("Next 30 days", f"{pf.forecast.head(30).sum():,.0f} units")
    c[2].metric("Next 90 days", f"{pf.forecast.sum():,.0f} units")
    top = pf.loc[pf.forecast.idxmax()]
    c[3].metric("Busiest day", f"{top.date:%a %d %b}", f"{top.forecast:.0f} units", delta_color="off")

    l, r = st.columns([1.3, 1])
    cal = pf.assign(week=pf.date - pd.to_timedelta(pf.date.dt.dayofweek, unit="D"), dow=pf.date.dt.dayofweek)
    piv = cal.pivot(index="week", columns="dow", values="forecast").reindex(columns=range(7))
    lab = cal.assign(lbl=cal.date.dt.strftime("%a %d %b")).pivot(index="week", columns="dow", values="lbl").reindex(columns=range(7))
    fig = go.Figure(go.Heatmap(z=piv.values, x=DOW, y=[f"wk of {w:%d %b}" for w in piv.index],
                               colorscale=[[0, "#FFF4EC"], [0.5, "#FFB27F"], [1, ORANGE]], xgap=3, ygap=3,
                               customdata=lab.fillna("").values, hovertemplate="%{customdata}: %{z:.0f} units<extra></extra>",
                               colorbar=dict(title="units")))
    fig.update_layout(height=430, title="Calendar view (darker = busier)", yaxis_autorange="reversed")
    l.plotly_chart(fig, width="stretch")
    dowm = pf.assign(d=pf.date.dt.dayofweek).groupby("d").forecast.mean().reindex(range(7))
    fig = go.Figure(go.Bar(x=DOW, y=dowm.values, marker_color=[ORANGE] * 5 + [PURPLE] * 2,
                           text=dowm.round(0).astype(int), textposition="outside"))
    fig.update_layout(height=430, title="Average forecast by weekday", yaxis_title="Units / day", yaxis_range=[0, dowm.max() * 1.2])
    r.plotly_chart(fig, width="stretch")

# =================================================================================== 3. PROMO PLANNER
with tabs[2]:
    st.markdown(f"#### 🏷️ Plan a discount on {name_of.get(sku, sku)}")
    st.caption("Pick dates and a discount. The model re-forecasts every day, so you see the extra units and whether revenue actually goes up.")
    c1, c2, c3 = st.columns(3)
    p_start = pd.Timestamp(c1.date_input("Promo starts", value=(first + pd.Timedelta(days=4)).date(), min_value=first.date(),
                                         max_value=last_f.date(), key="promo_start"))
    p_len = c2.slider("Promo length (days)", 1, 14, 3, key="promo_len")
    p_disc = c3.slider("Discount", 5, 50, 20, 5, format="%d%%", key="promo_disc")
    win = ((pf.date >= p_start) & (pf.date < p_start + pd.Timedelta(days=p_len))).values
    base_df, scen_df = with_conditions(pf), with_conditions(pf, win, p_disc / 100)
    b_units, s_units = predict(base_df), predict(scen_df)
    ext = ((pf.date >= p_start - pd.Timedelta(days=7)) & (pf.date < p_start + pd.Timedelta(days=p_len + 7))).values
    fig = go.Figure()
    fig.add_vrect(x0=p_start - pd.Timedelta(hours=12), x1=p_start + pd.Timedelta(days=p_len - 0.5),
                  fillcolor=YELLOW, opacity=0.18, line_width=0, annotation_text=f"🏷️ {p_disc}% off", annotation_position="top left")
    fig.add_bar(x=pf.date[ext], y=b_units[ext], name="without promo", marker_color=LIGHT)
    fig.add_bar(x=pf.date[ext], y=s_units[ext], name="with promo", marker_color=ORANGE)
    fig.update_layout(barmode="group", height=380, yaxis_title="Units / day", title="Daily forecast around the promotion")
    st.plotly_chart(fig, width="stretch")
    bu, su = b_units[win].sum(), s_units[win].sum()
    brev = (b_units[win] * base_df.selling_price.values[win]).sum()
    srev = (s_units[win] * scen_df.selling_price.values[win]).sum()
    c = st.columns(4)
    c[0].metric("Units during promo", f"{su:,.0f}", f"{su - bu:+,.0f} ({(su / bu - 1):+.0%})" if bu else None)
    c[1].metric("Revenue during promo", f"₹{srev:,.0f}", f"₹{srev - brev:+,.0f}")
    c[2].metric("Price per unit", f"₹{scen_df.selling_price.values[win][0]:,.0f}" if win.any() else "—",
                f"was ₹{pf.regular_price.iloc[0]:,.0f}", delta_color="off")
    verdict = ("✅ More units <b>and</b> more revenue — a good promo." if srev > brev and su > bu else
               "⚠️ More units but <b>less</b> revenue — the discount costs more than the extra sales bring in." if su > bu else
               "❌ No gain from this promo.")
    c[3].markdown(f"<div class='card'><b>Verdict</b><br>{verdict}</div>", unsafe_allow_html=True)
    measured = promo.set_index("category").uplift_pct.get(cat)
    if measured is not None:
        st.caption(f"In the historical data, {cat} sold {measured:.0f}% more units on promo days (~20% off).")

# =================================================================================== 4. SALES HISTORY
with tabs[3]:
    st.markdown(f"#### 📜 Sales history — {name_of.get(sku, sku)} at {store_name.get(store, store)}")
    dmin, dmax = ph.date.min().date(), ph.date.max().date()
    default_start = max(dmin, (ph.date.max() - pd.Timedelta(days=120)).date())
    rng = st.slider("Date range", min_value=dmin, max_value=dmax, value=(default_start, dmax), format="DD MMM YYYY", key="h_rng")
    t1, t2, t3, t4 = st.columns(4)
    sw = t1.toggle("🎈 weekends", True, key="h_w")
    sf = t2.toggle("🎉 festivals", True, key="h_f")
    sp = t3.toggle("🏷️ promos", True, key="h_p")
    so = t4.toggle("❌ sold out", True, key="h_o")
    v = ph[(ph.date >= pd.Timestamp(rng[0])) & (ph.date <= pd.Timestamp(rng[1]))]
    fig = go.Figure()
    if sp:
        for d in v[v.on_promo == 1].date:
            fig.add_vrect(x0=d - pd.Timedelta(hours=12), x1=d + pd.Timedelta(hours=12), fillcolor=YELLOW, opacity=0.2, line_width=0)
    fig.add_scatter(x=v.date, y=v.units_sold, name="units sold", line=dict(color=ORANGE, width=1.8),
                    hovertemplate="%{x|%a %d %b %Y}: %{y} units<extra></extra>")
    fig.add_scatter(x=v.date, y=v.units_sold.rolling(7, min_periods=1).mean(), name="7-day average",
                    line=dict(color=INK, width=2, dash="dot"))
    if sw:
        w = v[v.date.dt.dayofweek >= 5]
        fig.add_scatter(x=w.date, y=w.units_sold, mode="markers", name="weekend", marker=dict(color=PURPLE, size=7))
    if sf:
        f = v[v.is_festival == 1]
        fig.add_scatter(x=f.date, y=f.units_sold, mode="markers+text", name="festival", text=f.festival_name,
                        textposition="top center", marker=dict(color=MAGENTA, size=13, symbol="star"),
                        textfont=dict(color=MAGENTA, size=10))
    if so:
        o = v[v.stockout_flag == 1]
        fig.add_scatter(x=o.date, y=o.units_sold, mode="markers", name="sold out", marker=dict(color="#E53935", size=10, symbol="x"))
    fig.update_layout(height=430, hovermode="closest", yaxis_title="Units / day")
    st.plotly_chart(fig, width="stretch")

    clean = v[v.stockout_flag == 0]
    wkd, wke = clean[clean.date.dt.dayofweek < 5].units_sold.mean(), clean[clean.date.dt.dayofweek >= 5].units_sold.mean()
    nrm = clean[(clean.is_festival == 0) & (clean.on_promo == 0)].units_sold.mean()
    fes = clean[clean.is_festival == 1].units_sold.mean()
    prm = clean[(clean.on_promo == 1) & (clean.is_festival == 0)].units_sold.mean()

    def pct(a, b):
        return f"{(a / b - 1):+.0%} vs normal days" if pd.notna(a) and pd.notna(b) and b else None
    c = st.columns(4)
    c[0].metric("Weekday average", f"{wkd:.0f} units" if pd.notna(wkd) else "—")
    c[1].metric("Weekend average", f"{wke:.0f} units" if pd.notna(wke) else "—",
                f"{(wke / wkd - 1):+.0%} vs weekdays" if pd.notna(wke) and pd.notna(wkd) and wkd else None)
    c[2].metric("Festival-day average", f"{fes:.0f} units" if pd.notna(fes) else "no festival in range", pct(fes, nrm))
    c[3].metric("Promo-day average", f"{prm:.0f} units" if pd.notna(prm) else "no promo in range", pct(prm, nrm))
    st.caption(f"{len(v)} days shown · {int(v.stockout_flag.sum())} sold-out days (averages exclude them, because sales were capped by stock).")

    l, r = st.columns(2)
    mon = ph.assign(m=ph.date.dt.to_period("M").dt.to_timestamp()).groupby("m").units_sold.sum()
    fig = go.Figure(go.Bar(x=mon.index, y=mon.values, marker_color=ORANGE))
    fig.update_layout(height=320, title="Units sold per month (full history)", xaxis_tickformat="%b %y")
    l.plotly_chart(fig, width="stretch")
    nos = ph[ph.stockout_flag == 0]
    dw = nos.groupby(nos.date.dt.dayofweek).units_sold.mean().reindex(range(7))
    fig = go.Figure(go.Bar(x=DOW, y=dw.values, marker_color=[ORANGE] * 5 + [PURPLE] * 2, text=dw.round(0).astype(int),
                           textposition="outside"))
    fig.update_layout(height=320, title="Average units by weekday (full history)", yaxis_range=[0, dw.max() * 1.2])
    r.plotly_chart(fig, width="stretch")

# =================================================================================== 5. ORDER PLAN
with tabs[4]:
    day1 = fut[(fut.horizon_day == 1) & (fut.store_id == store)].copy()
    st.markdown(f"#### 📦 Stock plan for {store_name.get(store, store)} — {day1.date.iloc[0]:%A %d %b %Y}")
    st.caption("Target stock = tomorrow's forecast + safety stock. Safety stock grows with how important the product is "
               "(ABC class) and how unpredictable it is. Order = target stock − what's on the shelf tonight.")
    c1, c2, c3 = st.columns(3)
    zmap = {"A": norm.ppf(c1.slider("A products (top sellers)", 0.90, 0.995, 0.98, key="zA")),
            "B": norm.ppf(c2.slider("B products", 0.85, 0.99, 0.95, key="zB")),
            "C": norm.ppf(c3.slider("C products", 0.80, 0.98, 0.90, key="zC"))}
    day1["product"] = day1.category.map(emoji) + " " + day1.sku_id.map(lambda s: name_of.get(s, s))
    day1["forecast"] = day1.forecast.round(1)
    day1["safety_stock"] = np.ceil(day1.abc_class.map(zmap) * day1.sigma.fillna(0)).astype(int)
    day1["target_stock"] = np.ceil(day1.forecast + day1.safety_stock).astype(int)
    out = day1[["product", "category", "abc_class", "forecast", "safety_stock", "target_stock"]].sort_values(
        ["abc_class", "target_stock"], ascending=[True, False])
    c = st.columns(3)
    c[0].metric("Products", f"{len(out)}")
    c[1].metric("Total target stock", f"{out.target_stock.sum():,} units")
    c[2].metric("of which safety stock", f"{out.safety_stock.sum():,} units")
    l, r = st.columns([1.6, 1])
    l.dataframe(out, width="stretch", height=430, hide_index=True,
                column_config={"target_stock": st.column_config.ProgressColumn("target stock", format="%d", min_value=0,
                                                                               max_value=int(out.target_stock.max())),
                               "abc_class": "class", "safety_stock": "safety stock"})
    bycat = day1.groupby("category").target_stock.sum().sort_values()
    fig = go.Figure(go.Bar(y=[f"{emoji(c_)} {c_}" for c_ in bycat.index], x=bycat.values, orientation="h", marker_color=ORANGE))
    fig.update_layout(height=430, title="Target stock by category")
    r.plotly_chart(fig, width="stretch")
    st.download_button("⬇️ Download plan (CSV)", out.to_csv(index=False).encode("utf-8"),
                       file_name=f"stock_plan_{store}_{day1.date.iloc[0]:%Y%m%d}.csv", mime="text/csv")

# =================================================================================== 6. BUSINESS INSIGHTS
with tabs[5]:
    l, r = st.columns(2)
    fig = go.Figure()
    for c_, g in gmv.groupby("category"):
        fig.add_scatter(x=g.month, y=g.gmv / 1e5, name=f"{emoji(c_)} {c_}", mode="lines")
    fig.update_layout(height=420, title="Monthly sales value by category (₹ lakh)", legend=dict(y=-0.3))
    l.plotly_chart(fig, width="stretch")
    pr = promo.sort_values("uplift_pct")
    fig = go.Figure(go.Bar(y=[f"{emoji(c_)} {c_}" for c_ in pr.category], x=pr.uplift_pct, orientation="h", marker_color=MAGENTA,
                           text=[f"+{v:.0f}%" for v in pr.uplift_pct], textposition="outside"))
    fig.update_layout(height=420, title="Extra units sold on promo days (~20% off)", xaxis_range=[0, pr.uplift_pct.max() * 1.25])
    r.plotly_chart(fig, width="stretch")
    l, r = st.columns(2)
    o = oos.sort_values("est_lost_gmv_lakh")
    fig = go.Figure(go.Bar(y=["📍 " + s for s in o.store], x=o.est_lost_gmv_lakh, orientation="h", marker_color=PURPLE,
                           text=[f"₹{v:.1f}L" for v in o.est_lost_gmv_lakh], textposition="outside"))
    fig.update_layout(height=380, title="Revenue lost to stockouts, by store (₹ lakh)", xaxis_range=[0, o.est_lost_gmv_lakh.max() * 1.25])
    l.plotly_chart(fig, width="stretch")
    seg = abc.groupby(["abc_class", "xyz_class"]).revenue.sum().unstack().reindex(index=list("ABC"), columns=list("XYZ")).fillna(0)
    seg = 100 * seg / seg.values.sum()
    fig = go.Figure(go.Heatmap(z=seg.values, x=["X steady", "Y variable", "Z erratic"], y=["A top sellers", "B middle", "C tail"],
                               colorscale=[[0, "#FFF4EC"], [1, ORANGE]], text=np.round(seg.values, 0), texttemplate="%{text:.0f}%",
                               xgap=4, ygap=4, showscale=False))
    fig.update_layout(height=380, title="Share of revenue by product segment", yaxis_autorange="reversed")
    r.plotly_chart(fig, width="stretch")

# =================================================================================== 7. MODEL REPORT
with tabs[6]:
    st.markdown("#### 🧠 How good are the forecasts?")
    c = st.columns(4)
    c[0].metric("14-day model error", f"{M['wmape_lgbm_full']:.1%}", f"naive guess: {M['wmape_seasonal_naive']:.1%}", delta_color="off")
    c[1].metric("90-day model error", f"{M.get('wmape_long_range_90d', float('nan')):.1%}",
                f"same day last year: {M.get('wmape_same_day_last_year', float('nan')):.1%}", delta_color="off")
    c[2].metric("Promo-day error", f"{M['promo_day_wmape_full']:.1%}", f"without promo features: {M['promo_day_wmape_no_drivers']:.1%}",
                delta_color="off")
    c[3].metric("Forecast bias", f"{M['bias_lgbm_full_pct']:+.1f}%", "≈0 means no systematic over/under", delta_color="off")
    cm = comp.sort_values("wmape", ascending=False)
    fig = go.Figure(go.Bar(y=cm.model, x=cm.wmape * 100, orientation="h",
                           marker_color=[ORANGE if mm == "LightGBM, full" else LIGHT for mm in cm.model],
                           text=[f"{v:.1f}%" for v in cm.wmape * 100], textposition="outside"))
    fig.update_layout(height=340, title="Error of each method on 3 hidden 14-day periods (lower is better)",
                      xaxis_range=[0, cm.wmape.max() * 125])
    st.plotly_chart(fig, width="stretch")
    g = bt[(bt.store_id == store) & (bt.sku_id == sku)].sort_values("date")
    fig = go.Figure()
    fig.add_scatter(x=g.date, y=g.demand, name="actual", line=dict(color=INK))
    fig.add_scatter(x=g.date, y=g.p_seasonal_naive, name="naive guess", line=dict(color=LIGHT))
    fig.add_scatter(x=g.date, y=g.p_lgbm_full, name="model", line=dict(color=ORANGE, width=3))
    fig.update_layout(height=360, hovermode="x unified", title=f"Back-test for {name_of.get(sku, sku)}: model vs reality on hidden weeks")
    st.plotly_chart(fig, width="stretch")
    l, r = st.columns(2)
    l.image(str(R / "figures" / "09_feature_importance.png"))
    r.image(str(R / "figures" / "10_inventory_simulation.png"))
    st.caption("Built on simulated quick-commerce data with real-world patterns. The same pipeline runs on the Kaggle M5 dataset (see README).")
