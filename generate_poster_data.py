"""
Generate real data for the research poster figures:
  Figure 1: Monthly sentiment trends (Political Actors, News Outlets, General Public)
  Figure 2: Before vs During crisis sentiment comparison
  Figure 3: President vs Opposition sentiment trends

Uses scraped data from spain_politicians_data/
"""

import pandas as pd
import numpy as np
import os
import json

OUTPUT_DIR = "spain_politicians_data"

# ─── Crisis periods ──────────────────────────────────────────────────────────
# Real data shows: wildfires discourse in June, blackout discourse in Nov
CRISIS_MONTHS = [6, 10, 11]  # Wildfires (June), Blackout (Oct-Nov)
BEFORE_CRISIS_MONTHS = [1, 2, 3, 4, 5, 7, 8, 9, 12]

# ─── Politician classification ───────────────────────────────────────────────
PRESIDENT = ["Pedro Sanchez"]
OPPOSITION = {
    "Opp 1 (PP)":   ["Alberto Nunez Feijoo", "Isabel Diaz Ayuso"],
    "Opp 2 (Vox)":  ["Santiago Abascal"],
    "Opp 3 (Left)": ["Pablo Iglesias", "Irene Montero", "Ione Belarra"],
}
ALL_OPPOSITION = [n for names in OPPOSITION.values() for n in names]

# ─── Public period → month mapping ───────────────────────────────────────────
PUBLIC_PERIOD_TO_CRISIS = {
    "wildfires_june": "during",
    "blackout_oct": "during",
    "baseline_feb": "before",
    "baseline_apr": "before",
    "baseline_aug": "before",
    "baseline_dec": "before",
}
PUBLIC_PERIOD_MONTH = {
    "wildfires_june": 6,
    "blackout_oct": 10,
    "baseline_feb": 2,
    "baseline_apr": 4,
    "baseline_aug": 8,
    "baseline_dec": 12,
}
# ─────────────────────────────────────────────────────────────────────────────


def load_and_parse(csv_path):
    """Load CSV and parse dates into month."""
    df = pd.read_csv(csv_path)
    df["parsed_date"] = pd.to_datetime(
        df["date"], format="%a %b %d %H:%M:%S %z %Y", utc=True, errors="coerce")
    df["month"] = df["parsed_date"].dt.month
    df["year"] = df["parsed_date"].dt.year
    df = df[df["year"] == 2025].copy()
    return df


def load_public():
    """Load general public data with period-based month mapping."""
    path = os.path.join(OUTPUT_DIR, "general_public_2025.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["parsed_date"] = pd.to_datetime(
        df["date"], format="%a %b %d %H:%M:%S %z %Y", utc=True, errors="coerce")
    df["month"] = df["parsed_date"].dt.month
    df["is_crisis"] = df["period"].map(PUBLIC_PERIOD_TO_CRISIS)
    df["period_month"] = df["period"].map(PUBLIC_PERIOD_MONTH)
    return df


def generate_figure1():
    """Figure 1: Monthly sentiment trends for 3 actor groups."""
    print("\n" + "="*60)
    print("FIGURE 1: Monthly Sentiment Trends During Major Crises, Spain, 2025")
    print("="*60)

    results = {}

    # 1. Political Actors
    pol_path = os.path.join(OUTPUT_DIR, "all_politicians_2025.csv")
    if os.path.exists(pol_path):
        df = load_and_parse(pol_path)
        monthly = df.groupby("month")["sentiment"].mean()
        results["Political Actors"] = monthly
        print(f"\nPolitical Actors ({len(df)} tweets, {df['month'].nunique()} months):")
        print(monthly.round(3).to_string())

    # 2. News Outlets
    news_path = os.path.join(OUTPUT_DIR, "news_outlets_2025.csv")
    if os.path.exists(news_path):
        df = load_and_parse(news_path)
        monthly = df.groupby("month")["sentiment"].mean()
        results["News Outlets"] = monthly
        print(f"\nNews Outlets ({len(df)} tweets, {df['month'].nunique()} months):")
        print(monthly.round(3).to_string())

    # 3. General Public
    pub = load_public()
    if pub is not None:
        monthly = pub.groupby("month")["sentiment"].mean()
        results["General Public"] = monthly
        print(f"\nGeneral Public ({len(pub)} tweets, {pub['month'].nunique()} months):")
        print(monthly.round(3).to_string())

    # Build figure data table
    months = list(range(1, 13))
    fig1 = pd.DataFrame({"month": months})
    for group, series in results.items():
        fig1[group] = fig1["month"].map(series)

    fig1_path = os.path.join(OUTPUT_DIR, "figure1_monthly_sentiment.csv")
    fig1.to_csv(fig1_path, index=False)
    print(f"\n--- Figure 1 Data Table ---")
    print(fig1.round(3).to_string(index=False))
    print(f"\nSaved -> {fig1_path}")

    # Highlight crisis months
    for m in CRISIS_MONTHS:
        row = fig1[fig1["month"] == m]
        crisis_name = "Wildfires" if m == 6 else "Power Outage"
        print(f"\n  *** {crisis_name} (month {m}) ***")
        for col in fig1.columns[1:]:
            val = row[col].values[0] if not row[col].isna().values[0] else "N/A"
            print(f"    {col}: {val:.3f}" if isinstance(val, float) else f"    {col}: {val}")

    return fig1


def generate_figure2():
    """Figure 2: Before vs During crisis sentiment for 3 groups."""
    print("\n" + "="*60)
    print("FIGURE 2: Sentiment Before and During Crisis Periods, Spain, 2025")
    print("="*60)

    results = []

    # Political Actors
    pol_path = os.path.join(OUTPUT_DIR, "all_politicians_2025.csv")
    if os.path.exists(pol_path):
        df = load_and_parse(pol_path)
        before = df[df["month"].isin(BEFORE_CRISIS_MONTHS)]["sentiment"]
        during = df[df["month"].isin(CRISIS_MONTHS)]["sentiment"]
        results.append({
            "group": "Political Actors",
            "before_crisis": round(before.mean(), 3),
            "during_crisis": round(during.mean(), 3),
            "change": round(during.mean() - before.mean(), 3),
            "n_before": len(before), "n_during": len(during),
        })

    # News Outlets
    news_path = os.path.join(OUTPUT_DIR, "news_outlets_2025.csv")
    if os.path.exists(news_path):
        df = load_and_parse(news_path)
        before = df[df["month"].isin(BEFORE_CRISIS_MONTHS)]["sentiment"]
        during = df[df["month"].isin(CRISIS_MONTHS)]["sentiment"]
        results.append({
            "group": "News Outlets",
            "before_crisis": round(before.mean(), 3),
            "during_crisis": round(during.mean(), 3),
            "change": round(during.mean() - before.mean(), 3),
            "n_before": len(before), "n_during": len(during),
        })

    # General Public (use period tags for before/during)
    pub = load_public()
    if pub is not None:
        before = pub[pub["is_crisis"] == "before"]["sentiment"]
        during = pub[pub["is_crisis"] == "during"]["sentiment"]
        results.append({
            "group": "General Public",
            "before_crisis": round(before.mean(), 3),
            "during_crisis": round(during.mean(), 3),
            "change": round(during.mean() - before.mean(), 3),
            "n_before": len(before), "n_during": len(during),
        })

    fig2 = pd.DataFrame(results)
    fig2_path = os.path.join(OUTPUT_DIR, "figure2_crisis_comparison.csv")
    fig2.to_csv(fig2_path, index=False)

    print(f"\n--- Figure 2 Data Table ---")
    for _, row in fig2.iterrows():
        print(f"  {row['group']}:")
        print(f"    Before Crisis: {row['before_crisis']:.3f} (n={row['n_before']})")
        print(f"    During Crisis: {row['during_crisis']:.3f} (n={row['n_during']})")
        print(f"    Change: {row['change']:+.3f}")
    print(f"\nSaved -> {fig2_path}")
    return fig2


def generate_figure3():
    """Figure 3: President vs Opposition monthly sentiment."""
    print("\n" + "="*60)
    print("FIGURE 3: President vs Opposition Sentiment, Spain, 2025")
    print("="*60)

    pol_path = os.path.join(OUTPUT_DIR, "all_politicians_2025.csv")
    if not os.path.exists(pol_path):
        print("No politician data!")
        return None

    df = load_and_parse(pol_path)

    # President
    pres = df[df["politician"].isin(PRESIDENT)]
    pres_monthly = pres.groupby("month")["sentiment"].mean()

    # Opposition subgroups
    opp_monthly = {}
    for label, names in OPPOSITION.items():
        opp = df[df["politician"].isin(names)]
        if not opp.empty:
            opp_monthly[label] = opp.groupby("month")["sentiment"].mean()

    # Overall opposition average
    all_opp = df[df["politician"].isin(ALL_OPPOSITION)]
    opp_avg_monthly = all_opp.groupby("month")["sentiment"].mean()

    months = list(range(1, 13))
    fig3 = pd.DataFrame({"month": months})
    fig3["President"] = fig3["month"].map(pres_monthly)
    fig3["Opp Avg"] = fig3["month"].map(opp_avg_monthly)
    for label, series in opp_monthly.items():
        fig3[label] = fig3["month"].map(series)

    fig3_path = os.path.join(OUTPUT_DIR, "figure3_president_vs_opposition.csv")
    fig3.to_csv(fig3_path, index=False)

    print(f"\n--- Figure 3 Data Table ---")
    print(fig3.round(3).to_string(index=False))
    print(f"\nSaved -> {fig3_path}")

    # Key finding analysis
    print(f"\n--- Key Finding: President goes positive in crisis, opposition criticizes ---")
    for m in CRISIS_MONTHS:
        crisis_name = "Wildfires" if m == 6 else "Power Outage"
        p = fig3.loc[fig3["month"] == m, "President"].values
        o = fig3.loc[fig3["month"] == m, "Opp Avg"].values
        pv = f"{p[0]:.3f}" if len(p) and not np.isnan(p[0]) else "N/A"
        ov = f"{o[0]:.3f}" if len(o) and not np.isnan(o[0]) else "N/A"
        print(f"  {crisis_name} (month {m}): President={pv}, Opp Avg={ov}")

    # Non-crisis average
    pres_non = fig3[fig3["month"].isin(BEFORE_CRISIS_MONTHS)]["President"].mean()
    opp_non = fig3[fig3["month"].isin(BEFORE_CRISIS_MONTHS)]["Opp Avg"].mean()
    print(f"\n  Non-crisis average: President={pres_non:.3f}, Opp Avg={opp_non:.3f}")

    return fig3


def generate_summary():
    """Overall summary statistics for the poster."""
    print("\n" + "="*60)
    print("OVERALL DATASET SUMMARY")
    print("="*60)

    stats = {}
    for group, csv_name in [
        ("Political Actors", "all_politicians_2025.csv"),
        ("News Outlets", "news_outlets_2025.csv"),
        ("General Public", "general_public_2025.csv"),
    ]:
        path = os.path.join(OUTPUT_DIR, csv_name)
        if os.path.exists(path):
            if group == "General Public":
                df = load_public()
            else:
                df = load_and_parse(path)
            stats[group] = {
                "total_tweets": len(df),
                "avg_sentiment": round(df["sentiment"].mean(), 3),
                "std_sentiment": round(df["sentiment"].std(), 3),
                "pct_negative": round((df["sentiment"] < -0.05).mean() * 100, 1),
                "pct_neutral": round(((df["sentiment"] >= -0.05) & (df["sentiment"] <= 0.05)).mean() * 100, 1),
                "pct_positive": round((df["sentiment"] > 0.05).mean() * 100, 1),
            }
            print(f"\n  {group}: {stats[group]['total_tweets']} tweets, "
                  f"avg sentiment={stats[group]['avg_sentiment']}, "
                  f"neg={stats[group]['pct_negative']}%, "
                  f"neu={stats[group]['pct_neutral']}%, "
                  f"pos={stats[group]['pct_positive']}%")

    total = sum(s["total_tweets"] for s in stats.values())
    print(f"\n  TOTAL TWEETS ANALYZED: {total}")

    stats_path = os.path.join(OUTPUT_DIR, "summary_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved -> {stats_path}")


if __name__ == "__main__":
    fig1 = generate_figure1()
    fig2 = generate_figure2()
    fig3 = generate_figure3()
    generate_summary()
