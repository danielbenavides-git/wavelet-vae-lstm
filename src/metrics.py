"""Metrics of the most expensive regime against the real prices and the El Nino days.

Every metric is computed on one period: all days, train, val or test. A day is expensive when its real price is above
the 66.7, 75 or 80 percentile (k = 3, 4 and 5) of the days of the same period. The 7d view uses the backward 7-day
mean of the price. A day is El Nino when the ONI of its month is >= 0.5.

Ranking: criterion A orders the configurations by expensive_f1_pct. Criterion C keeps the configurations that switch
at most persistence_factor times as often as the 7-day baseline and orders them by A. Models are chosen on val and
reported on test.

Calendar weeks use the same functions with one row per week and the weekly baseline (reference = "level").
"""
import numpy as np
import pandas as pd
from prettytable import PrettyTable

from src.clustering import regime_profile, run_lengths
from src.data import split_names
from src.runs import nino_episodes

thresholds = {3: 66.7, 4: 75.0, 5: 80.0}
price_views = {"daily": 1, "7d": 7, "week": 1}  # rows averaged backward; week: one row is already one week
days_per_year, weeks_per_year = 365.25, 365.25 / 7
periods = ["all", "train", "val", "test"]
persistence_factor = 2.0


def ep_key(name):
    """2015-2016 -> 2015_16"""
    return f"{name[:4]}_{name[7:9]}"


def days_from_context(context):
    """One row per day with date, real_price, nino (1 if ONI >= 0.5), episode (2015-2016, 2023-2024 or empty) and
    period (train, val or test)."""
    date = context["date"]
    nino = (context["enso_phase"] == "El Nino").astype(int)
    episode = pd.Series("", index=context.index)
    for name, years in nino_episodes.items():
        episode[(nino == 1) & date.dt.year.isin(years)] = name
    return pd.DataFrame({"date": date, "real_price": context["precio_real_cop_kwh"], "nino": nino,
                         "episode": episode, "period": split_names(date)}).reset_index(drop=True)


def expensive_days(days, k, view, period="all"):
    """Boolean array of the expensive days of period and the threshold in COP/kWh, with the daily price
    (daily view) or its backward 7-day mean (7d view)."""
    price = days.sort_values("date")["real_price"].rolling(price_views[view], min_periods=1).mean()
    price = price.reindex(days.index).to_numpy()
    keep = np.ones(len(days), bool) if period == "all" else days["period"].to_numpy() == period
    threshold = float(np.percentile(price[keep], thresholds[k]))
    return price > threshold, threshold


def data_reference(days, k_values, views=("daily", "7d")):
    """Level 1, no model. One row per (k, view, group); group is all days, one period or one episode. With weeks
    (views=("week",)) every row of days is a week and the counts are weeks."""
    nino = days["nino"].to_numpy() == 1
    in_episode = days["episode"].to_numpy() != ""
    price = days["real_price"].to_numpy()
    rows = []
    for k in k_values:
        for view in views:
            expensive_all = expensive_days(days, k, view)[0]
            groups = {"all": np.ones(len(days), bool),
                      **{period: days["period"].to_numpy() == period for period in periods[1:]},
                      **{name: days["episode"].to_numpy() == name for name in nino_episodes}}
            for group, mask in groups.items():
                # periods use their own threshold; the episodes use the one of all days
                expensive, threshold = expensive_days(days, k, view, group if group in periods else "all")
                rows.append({
                    "k": k, "threshold_pct": thresholds[k], "view": view, "threshold_cop_kwh": round(threshold, 1),
                    "group": group, "days": int(mask.sum()), "nino_days": int((mask & nino).sum()),
                    "expensive_pct": 100 * expensive[mask].mean(),
                    "median_price": float(np.median(price[mask])),
                    "median_price_rest": float(np.median(price[~in_episode])),
                    "base_expensive_pct": 100 * expensive_all.mean(), "base_nino_pct": 100 * nino.mean(),
                })
    return pd.DataFrame(rows).round(2)


def regime_metrics(days, k, view, period="all", units_per_year=days_per_year):
    """Metrics of the most expensive regime (the highest number) against expensive_days(k, view) on the days
    of period. units_per_year is days_per_year for days or weeks_per_year for calendar weeks."""
    expensive, threshold = expensive_days(days, k, view, period)
    top_regime = days["regime"].max()
    keep = np.ones(len(days), bool) if period == "all" else days["period"].to_numpy() == period
    days, expensive = days[keep], expensive[keep]
    top = days["regime"].to_numpy() == top_regime
    nino = days["nino"].to_numpy() == 1
    top_share = 100 * top.mean()
    out = {"view": view, "period": period, "threshold_cop_kwh": round(threshold, 1), "top_share_pct": top_share}
    for name in nino_episodes:
        ep, key = days["episode"].to_numpy() == name, ep_key(name)
        if not ep.any():
            for c in ("expensive", "nino_recall", "expensive_recall", "cheap_in_top"):
                out[f"{c}_{key}_pct"] = np.nan
            out[f"nino_lift_{key}"] = np.nan
            continue
        recall = 100 * top[ep].mean()
        out[f"expensive_{key}_pct"] = 100 * expensive[ep].mean()
        out[f"nino_recall_{key}_pct"] = recall
        out[f"nino_lift_{key}"] = recall / top_share
        out[f"expensive_recall_{key}_pct"] = 100 * top[ep & expensive].mean()
        out[f"cheap_in_top_{key}_pct"] = 100 * top[ep & ~expensive].mean() if (ep & ~expensive).any() else np.nan
    precision = 100 * nino[top].mean()
    out["top_nino_precision_pct"] = precision
    out["top_nino_lift"] = precision / (100 * nino.mean()) if nino.any() else np.nan
    out["top_expensive_precision_pct"] = 100 * expensive[top].mean()
    out["expensive_recall_pct"] = 100 * top[expensive].mean()
    p, r = out["top_expensive_precision_pct"], out["expensive_recall_pct"]
    out["expensive_f1_pct"] = 2 * p * r / (p + r) if p + r > 0 else 0.0
    runs = run_lengths(days.sort_values("date")["regime"].to_numpy())
    out["switches_per_year"] = (len(runs) - 1) / (len(days) / units_per_year)
    out["mean_run_days"] = float(runs.mean())
    return out


table_columns = {
    "switches_per_year": "Switches / yr", "mean_run_days": "Mean run (d)",
    "expensive_f1_pct": "Expensive F1 %",
    "top_share_pct": "Top %",
    "nino_recall_2015_16_pct": "Recall 15-16 %", "nino_lift_2015_16": "Lift 15-16",
    "nino_recall_2023_24_pct": "Recall 23-24 %", "nino_lift_2023_24": "Lift 23-24",
    "top_nino_lift": "Nino lift in top",
    "top_expensive_precision_pct": "Top expensive %",
    "expensive_recall_pct": "Expensive recall %",
    "cheap_in_top_2015_16_pct": "Cheap in top 15-16 %", "cheap_in_top_2023_24_pct": "Cheap in top 23-24 %",
}
config_columns = ["model", "study", "z_dim", "lam", "beta", "features"]


def ranking(rows, k, period="all", reference="level 7d"):
    """One row per configuration (mean and std over seeds) with its rank by criterion A and, if it passes the
    persistence filter, by criterion C. reference is the baseline of the filter: "level 7d", or "level" for calendar
    weeks."""
    data = rows[(rows["k"] == k) & rows["matched"] & (rows["period"] == period)].fillna({c: "-" for c in config_columns})
    keys = config_columns + ["view"]
    columns = list(table_columns)
    stats = data.groupby(keys, sort=False)[columns].agg(["mean", "std"])
    stats.columns = [f"{c}_{s}" for c, s in stats.columns]
    stats["seeds"] = data.groupby(keys, sort=False).size()
    stats = stats.reset_index()
    baseline = stats["model"] == "level_only"
    limit = persistence_factor * stats.loc[baseline & (stats["features"] == reference), "switches_per_year_mean"].iloc[0]
    stats["k"], stats["period"], stats["switches_limit"] = k, period, limit
    models = stats[~baseline].sort_values("expensive_f1_pct_mean", ascending=False)
    stats.loc[models.index, "rank_a"] = range(1, len(models) + 1)
    passed = models[models["switches_per_year_mean"] <= limit]
    stats.loc[passed.index, "rank_c"] = range(1, len(passed) + 1)
    return stats


ranking_columns = ["switches_per_year", "mean_run_days", "expensive_f1_pct", "top_expensive_precision_pct",
                   "expensive_recall_pct", "top_share_pct", "nino_lift_2015_16", "nino_lift_2023_24", "top_nino_lift"]


comparison_columns = {
    "switches_per_year": "Switches / yr", "top_share_pct": "Top regime days %",
    "nino_recall_2015_16_pct": "Capture 15-16 %", "nino_lift_2015_16": "Lift 15-16",
    "nino_recall_2023_24_pct": "Capture 23-24 %", "nino_lift_2023_24": "Lift 23-24",
    "top_expensive_precision_pct": "Precision %", "expensive_recall_pct": "Recall %", "expensive_f1_pct": "F1 %",
}


def seed_mean(rows):
    """Rows of one configuration run with several seeds as one row per view and period, with column_mean,
    column_std and seeds."""
    keys = [c for c in ("features", "view", "period") if c in rows.columns]
    columns = [c for c in comparison_columns if c in rows.columns]
    stats = rows.groupby(keys, sort=False)[columns].agg(["mean", "std"])
    stats.columns = [f"{c}_{s}" for c, s in stats.columns]
    first = rows.groupby(keys, sort=False).first().drop(columns=columns)
    return first.join(stats).assign(seeds=rows.groupby(keys, sort=False).size()).reset_index()


def comparison_table(rows, labels, unit="days"):
    """Table with the label columns (dict column -> header) and comparison_columns. unit names the rows in
    the headers: days, or weeks for calendar weeks."""
    headers = [h.replace("days", unit) for h in comparison_columns.values()]
    t = PrettyTable([*labels.values(), *headers])
    for _, r in rows.iterrows():
        cells = []
        for c in comparison_columns:
            digits = 2 if "lift" in c else 1
            value = r.get(c + "_mean")
            if pd.isna(value):  # a row that is not a mean over seeds (rows of several kinds can be mixed)
                value = r.get(c)
            if pd.isna(value):
                cells.append("-")
                continue
            cell = f"{value:.{digits}f}"
            if r.get("seeds", 1) > 1 and pd.notna(r.get(c + "_std")):
                cell += f" +- {r[c + '_std']:.{digits}f}"
            cells.append(cell)
        t.add_row([r[c] for c in labels] + cells)
    return t


def profile_rows(context, regimes, identity):
    """Regime profile (days, share, real price, reservoir, ONI, El Nino share of each regime) with the identity columns."""
    return regime_profile(context.assign(regime=regimes)).assign(**identity)
