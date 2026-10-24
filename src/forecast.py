"""
Stage 2 - demand forecasting with rolling-origin backtests.

Business framing: each dark store places replenishment plans 14 days ahead, so every
feature must be known at least 14 days before the target date (no leakage).

Models compared
  - Seasonal naive          : same weekday, 2 weeks earlier
  - 28-day moving average
  - Weekday profile         : mean of the same weekday over the last 4 available weeks
  - LightGBM (no drivers)   : lags/rolling stats + calendar only (ablation)
  - LightGBM (full)         : + promotions, price, festivals  -> global Tweedie model

Stockout days are CENSORED (sales < true demand). We impute demand on those days for
lag features and drop them from the training target, so the model learns demand, not sales.
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from data import FIG, REPORTS, ROOT, load_tables
from viz import MUTED, SERIES, plt, save

H = 14  # forecast horizon in days
N_FOLDS = 3


def wmape(y, p):
    return float(np.abs(y - p).sum() / np.abs(y).sum())


def bias(y, p):
    return float((p - y).sum() / y.sum())


def build_features(t):
    s = t["sales"].merge(t["products"][["sku_id", "category"]], on="sku_id").merge(t["calendar"], on="date", how="left")
    s = s.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)
    g = s.groupby(["store_id", "sku_id"], sort=False)

    # --- censored-demand correction: on stockout days demand >= sales; impute from recent in-stock days
    instock = s.units_sold.where(s.stockout_flag == 0)
    recent = instock.groupby([s.store_id, s.sku_id]).transform(lambda x: x.shift(1).rolling(7, min_periods=2).mean())
    s["demand"] = np.where(s.stockout_flag == 1, np.maximum(s.units_sold, recent.fillna(s.units_sold)), s.units_sold)

    d = g["demand"]
    for L in (14, 21, 28, 35, 42, 364):
        s[f"lag_{L}"] = d.shift(L)
    shifted = d.shift(H)
    sg = shifted.groupby([s.store_id, s.sku_id])
    for w in (7, 14, 28, 56):
        s[f"roll_mean_{w}"] = sg.transform(lambda x, w=w: x.rolling(w, min_periods=max(3, w // 2)).mean())
    s["roll_std_28"] = sg.transform(lambda x: x.rolling(28, min_periods=14).std())
    s["ewm_14"] = sg.transform(lambda x: x.ewm(alpha=0.1).mean())
    s["dow_mean_4w"] = s[["lag_14", "lag_21", "lag_28", "lag_35"]].mean(axis=1)
    s["trend_ratio"] = s["roll_mean_7"] / s["roll_mean_56"]
    s["stockout_rate_28"] = g["stockout_flag"].transform(lambda x: x.shift(H).rolling(28, min_periods=7).mean())

    # --- calendar (known in advance)
    s["dow"] = s.date.dt.dayofweek
    s["dom"] = s.date.dt.day
    s["month"] = s.date.dt.month
    s["weekofyear"] = s.date.dt.isocalendar().week.astype(int)
    s["is_weekend"] = (s.dow >= 5).astype(int)
    fest_dates = np.sort(s.loc[s.is_festival == 1, "date"].unique())
    dates = s.date.values
    idx = np.searchsorted(fest_dates, dates)
    nxt = np.where(idx < len(fest_dates), fest_dates[np.minimum(idx, len(fest_dates) - 1)], np.datetime64("NaT", "ns"))
    s["days_to_festival"] = ((nxt - dates) / np.timedelta64(1, "D")).clip(0, 60)
    prv = np.where(idx > 0, fest_dates[np.maximum(idx - 1, 0)], np.datetime64("NaT", "ns"))
    s["days_since_festival"] = ((dates - prv) / np.timedelta64(1, "D")).clip(0, 60)

    # --- planned commercial drivers (known when the promo calendar is set)
    s["price_index"] = s.selling_price / g["selling_price"].transform("max")
    for c in ("store_id", "sku_id", "category"):
        s[c + "_cat"] = s[c].astype("category")
    return s


BASE_FEATS = ["lag_14", "lag_21", "lag_28", "lag_35", "lag_42", "lag_364", "roll_mean_7", "roll_mean_14",
              "roll_mean_28", "roll_mean_56", "roll_std_28", "ewm_14", "dow_mean_4w", "trend_ratio",
              "stockout_rate_28", "dow", "dom", "month", "weekofyear", "is_weekend",
              "store_id_cat", "sku_id_cat", "category_cat"]
DRIVER_FEATS = ["on_promo", "discount_pct", "price_index", "is_festival", "diwali_window", "ipl_season",
                "days_to_festival", "days_since_festival"]
FULL_FEATS = BASE_FEATS + DRIVER_FEATS

PARAMS = dict(objective="tweedie", tweedie_variance_power=1.2, learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, verbose=-1, seed=7)


def fit_lgb(train, feats, rounds=700):
    tr = train[train.stockout_flag == 0]  # censored target excluded
    return lgb.train(PARAMS, lgb.Dataset(tr[feats], tr["demand"]), num_boost_round=rounds)


def backtest(s):
    end = s.date.max()
    folds, preds = [], []
    for k in range(N_FOLDS, 0, -1):
        test_start = end - pd.Timedelta(days=H * k - 1)
        test_end = test_start + pd.Timedelta(days=H - 1)
        train = s[(s.date < test_start) & s.lag_42.notna()]
        test = s[(s.date >= test_start) & (s.date <= test_end)].copy()
        test["fold"] = N_FOLDS - k + 1
        test["p_seasonal_naive"] = test["lag_14"]
        test["p_ma28"] = test["roll_mean_28"]
        test["p_weekday_profile"] = test["dow_mean_4w"]
        m_base = fit_lgb(train, BASE_FEATS)
        m_full = fit_lgb(train, FULL_FEATS)
        test["p_lgbm_no_drivers"] = m_base.predict(test[BASE_FEATS])
        test["p_lgbm_full"] = m_full.predict(test[FULL_FEATS])
        preds.append(test)
        print(f"fold {N_FOLDS - k + 1}: train to {train.date.max().date()}, test {test_start.date()} -> {test_end.date()}")
    return pd.concat(preds), m_full


MODELS = {"p_seasonal_naive": "Seasonal naive", "p_ma28": "28-day moving avg",
          "p_weekday_profile": "Weekday profile (4w)", "p_lgbm_no_drivers": "LightGBM, no drivers",
          "p_lgbm_full": "LightGBM, full"}


def evaluate(bt, abc):
    ev = bt[bt.stockout_flag == 0].merge(abc[["sku_id", "abc_class"]], on="sku_id")  # uncensored actuals only
    rows = []
    for col, name in MODELS.items():
        e = ev.dropna(subset=[col])
        rows.append({"model": name, "wmape": wmape(e.demand, e[col]), "bias": bias(e.demand, e[col]),
                     **{f"wmape_fold{f}": wmape(x.demand, x[col]) for f, x in e.groupby("fold")}})
    overall = pd.DataFrame(rows)
    by_cat = ev.groupby("category").apply(lambda x: pd.Series({
        "Seasonal naive": wmape(x.demand, x.p_seasonal_naive), "LightGBM, full": wmape(x.demand, x.p_lgbm_full)}),
        include_groups=False)
    promo = ev.groupby("on_promo").apply(lambda x: pd.Series({
        "LightGBM, no drivers": wmape(x.demand, x.p_lgbm_no_drivers), "LightGBM, full": wmape(x.demand, x.p_lgbm_full)}),
        include_groups=False)
    return overall, by_cat, promo


def simulate_inventory(bt, abc, pred_col, z_scale=1.0):
    """
    Daily order-up-to policy per store-SKU. Order at end of day t arrives start of day t+1 (lead time 1).
    Order-up-to level S_{t+1} = forecast_{t+1} + z * sigma, sigma = forecast error std from the
    calibration fold (fold 1); simulation runs on folds 2-3. Unmet demand is lost (no backorders).
    Service targets by ABC class: A 98%, B 95%, C 90%.
    """
    z_map = {"A": 2.054, "B": 1.645, "C": 1.282}
    cal = bt[bt.fold == 1]
    sigma = (cal.demand - cal[pred_col]).groupby([cal.store_id, cal.sku_id]).std().rename("sigma")
    sim = bt[bt.fold > 1].merge(sigma.reset_index(), on=["store_id", "sku_id"]).merge(abc[["sku_id", "abc_class"]], on="sku_id")
    sim = sim.sort_values(["store_id", "sku_id", "date"])
    sim["S"] = np.ceil(sim[pred_col].clip(lower=0) + z_scale * sim.abc_class.map(z_map) * sim.sigma)
    out = []
    for _, g in sim.groupby(["store_id", "sku_id"], sort=False):
        S, D = g.S.values, g.demand.values
        on_hand = S[0]
        sold = np.zeros(len(g)); inv = np.zeros(len(g)); oos = np.zeros(len(g))
        for i in range(len(g)):
            sold[i] = min(D[i], on_hand)
            oos[i] = D[i] > on_hand
            on_hand -= sold[i]
            inv[i] = on_hand  # closing stock (what sits on the shelf overnight)
            if i + 1 < len(g):
                on_hand = max(on_hand, S[i + 1])  # replenish up to tomorrow's target
        out.append((D.sum(), sold.sum(), oos.sum(), inv.sum(), len(g)))
    a = np.array(out)
    demand, sold, oos, inv, days = a.sum(axis=0)
    return {"fill_rate": sold / demand, "stockout_day_rate": oos / days,
            "avg_closing_stock_units": inv / days, "days_of_cover": (inv / days) / (demand / days)}


def iso_service_comparison(bt, abc, target_fill):
    """Scale the baseline's safety stock until it matches the ML policy's fill rate, then compare inventory."""
    lo, hi = 0.5, 6.0
    for _ in range(18):
        mid = (lo + hi) / 2
        if simulate_inventory(bt, abc, "p_seasonal_naive", mid)["fill_rate"] < target_fill:
            lo = mid
        else:
            hi = mid
    return hi, simulate_inventory(bt, abc, "p_seasonal_naive", hi)


def iso_inventory_comparison(bt, abc, target_cover):
    """Scale the ML policy's safety stock until it holds the SAME stock as the baseline, then compare stockouts."""
    lo, hi = 0.5, 8.0
    for _ in range(18):
        mid = (lo + hi) / 2
        if simulate_inventory(bt, abc, "p_lgbm_full", mid)["days_of_cover"] < target_cover:
            lo = mid
        else:
            hi = mid
    return hi, simulate_inventory(bt, abc, "p_lgbm_full", hi)


def charts(bt, overall, by_cat, model, inv):
    # Model comparison
    o = overall.sort_values("wmape", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = [SERIES[0] if m == "LightGBM, full" else "#b9b8b2" for m in o.model]
    ax.barh(o.model, o.wmape * 100, color=colors, height=0.6)
    for y, v in enumerate(o.wmape * 100):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("WMAPE across 3 rolling backtests, % (lower is better)")
    ax.set_title("14-day-ahead forecast accuracy")
    ax.set_xlim(0, o.wmape.max() * 118)
    save(fig, FIG / "06_model_comparison.png")

    # Category accuracy
    c = by_cat.sort_values("LightGBM, full") * 100
    fig, ax = plt.subplots(figsize=(7.5, 4))
    y = np.arange(len(c))
    ax.barh(y + 0.18, c["Seasonal naive"], height=0.34, color="#b9b8b2", label="Seasonal naive")
    ax.barh(y - 0.18, c["LightGBM, full"], height=0.34, color=SERIES[0], label="LightGBM, full")
    ax.set_yticks(y, c.index)
    ax.set_xlabel("WMAPE, %")
    ax.set_title("Forecast error by category")
    ax.legend(loc="lower right")
    save(fig, FIG / "07_wmape_by_category.png")

    # Forecast vs actual for the highest-volume store-SKU
    top = bt.groupby(["store_id", "sku_id"]).demand.sum().idxmax()
    g = bt[(bt.store_id == top[0]) & (bt.sku_id == top[1])].sort_values("date")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(g.date, g.demand, color="#0b0b0b", label="Actual demand", lw=1.6)
    ax.plot(g.date, g.p_seasonal_naive, color="#b9b8b2", label="Seasonal naive", lw=1.6)
    ax.plot(g.date, g.p_lgbm_full, color=SERIES[0], label="LightGBM, full")
    promo_days = g[g.on_promo == 1].date
    for d in promo_days:
        ax.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12), color=SERIES[3], alpha=0.15, lw=0)
    ax.set_title(f"Backtest: {top[1]} at {top[0]} (shaded = promo days)")
    ax.set_ylabel("Units / day")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.1))
    save(fig, FIG / "08_forecast_vs_actual.png")

    # Feature importance (gain)
    imp = pd.Series(model.feature_importance("gain"), index=model.feature_name()).sort_values().tail(15)
    imp = imp / imp.sum() * 100
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(imp.index, imp.values, color=SERIES[0], height=0.6)
    ax.set_xlabel("Share of total gain, %")
    ax.set_title("What drives the forecast (LightGBM gain)")
    save(fig, FIG / "09_feature_importance.png")

    # Inventory policy comparison
    grey, blue = "#b9b8b2", SERIES[0]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    # Panel A: same fill rate -> stock held
    a_vals = [inv["naive_iso"]["days_of_cover"], inv["lgbm"]["days_of_cover"]]
    axs[0].bar(["Seasonal-naive policy", "LightGBM policy"], a_vals, color=[grey, blue], width=0.5)
    for x, v in enumerate(a_vals):
        axs[0].text(x, v, f"{v:.2f} days", ha="center", va="bottom", fontsize=9)
    axs[0].set_title(f"Same {inv['lgbm']['fill_rate']:.1%} fill rate → overnight stock held")
    axs[0].set_ylabel("Days of cover")
    # Panel B: same stock -> stockout days
    b_vals = [inv["naive"]["stockout_day_rate"] * 100, inv["lgbm_iso_inv"]["stockout_day_rate"] * 100]
    axs[1].bar(["Seasonal-naive policy", "LightGBM policy"], b_vals, color=[grey, blue], width=0.5)
    for x, v in enumerate(b_vals):
        axs[1].text(x, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
    axs[1].set_title("Same stock held → SKU-days with a stockout")
    axs[1].set_ylabel("% of SKU-days")
    fig.suptitle("Replenishment simulation on 28 held-out days", x=0.02, ha="left", fontweight="bold")
    save(fig, FIG / "10_inventory_simulation.png")


def main():
    t = load_tables()
    abc = pd.read_csv(REPORTS / "sql_abc_xyz.csv")
    print("building features ...")
    s = build_features(t)
    bt, _ = backtest(s)
    overall, by_cat, promo = evaluate(bt, abc)
    print(overall.round(4).to_string(index=False))

    inv = {"naive": simulate_inventory(bt, abc, "p_seasonal_naive"),
           "lgbm": simulate_inventory(bt, abc, "p_lgbm_full")}
    scale, inv["naive_iso"] = iso_service_comparison(bt, abc, inv["lgbm"]["fill_rate"])
    _, inv["lgbm_iso_inv"] = iso_inventory_comparison(bt, abc, inv["naive"]["days_of_cover"])
    print(pd.DataFrame(inv).round(4))

    # final model on all history for deployment / the dashboard
    final = fit_lgb(s[s.lag_42.notna()], FULL_FEATS)
    (ROOT / "models").mkdir(exist_ok=True)
    final.save_model(str(ROOT / "models" / "lgbm_demand.txt"))
    joblib.dump({"features": FULL_FEATS, "horizon": H}, ROOT / "models" / "meta.joblib")
    charts(bt, overall, by_cat, final, inv)

    keep = ["date", "store_id", "sku_id", "category", "fold", "on_promo", "stockout_flag", "demand", *MODELS]
    bt[keep].to_parquet(REPORTS / "backtest_predictions.parquet", index=False)
    overall.to_csv(REPORTS / "model_comparison.csv", index=False)
    by_cat.to_csv(REPORTS / "wmape_by_category.csv")

    best = overall.set_index("model")
    naive_w, full_w, nod_w = (best.loc[m, "wmape"] for m in ("Seasonal naive", "LightGBM, full", "LightGBM, no drivers"))
    summary = {
        "horizon_days": H, "folds": N_FOLDS,
        "wmape_seasonal_naive": round(naive_w, 4), "wmape_ma28": round(best.loc["28-day moving avg", "wmape"], 4),
        "wmape_lgbm_no_drivers": round(nod_w, 4), "wmape_lgbm_full": round(full_w, 4),
        "wmape_reduction_vs_naive_pct": round(100 * (1 - full_w / naive_w), 1),
        "bias_lgbm_full_pct": round(100 * best.loc["LightGBM, full", "bias"], 2),
        "promo_day_wmape_no_drivers": round(float(promo.loc[1, "LightGBM, no drivers"]), 4),
        "promo_day_wmape_full": round(float(promo.loc[1, "LightGBM, full"]), 4),
        "fill_rate_naive": round(inv["naive"]["fill_rate"], 4), "fill_rate_lgbm": round(inv["lgbm"]["fill_rate"], 4),
        # Fair comparison 1: same stock held -> how many fewer stockout days?
        "stockout_days_naive": round(inv["naive"]["stockout_day_rate"], 4),
        "stockout_days_lgbm_same_inventory": round(inv["lgbm_iso_inv"]["stockout_day_rate"], 4),
        "stockout_reduction_at_same_inventory_pct": round(100 * (1 - inv["lgbm_iso_inv"]["stockout_day_rate"] / inv["naive"]["stockout_day_rate"]), 1),
        "lost_units_reduction_at_same_inventory_pct": round(100 * (1 - (1 - inv["lgbm_iso_inv"]["fill_rate"]) / (1 - inv["naive"]["fill_rate"])), 1),
        # Fair comparison 2: same fill rate -> how much less stock?
        "days_cover_naive_iso": round(inv["naive_iso"]["days_of_cover"], 3), "days_cover_lgbm": round(inv["lgbm"]["days_of_cover"], 3),
        "inventory_reduction_at_same_fill_pct": round(100 * (1 - inv["lgbm"]["days_of_cover"] / inv["naive_iso"]["days_of_cover"]), 1),
    }
    (REPORTS / "model_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
