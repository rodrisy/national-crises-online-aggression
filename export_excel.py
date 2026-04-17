"""
Export all poster data into clean Excel spreadsheets,
ready for manual chart creation in Excel.
"""

import pandas as pd
import numpy as np
import os
import re
from collections import Counter

OUTPUT_DIR = "spain_politicians_data"

# ─── Keyword lists (same as analysis) ────────────────────────────────────────
AGGRESSION_WORDS = [
    "vergüenza", "vergonzoso", "vergonzosa", "inútil", "inútiles",
    "incompetente", "incompetentes", "inepto", "ineptos",
    "mentira", "mentiras", "mentiroso", "mentirosa", "miente",
    "corrupto", "corrupta", "corrupción", "ladrón", "ladrones",
    "estafa", "estafador", "delincuente", "criminal", "criminales",
    "miserable", "sinvergüenza", "hipócrita", "hipócritas",
    "cobarde", "cobardes", "traidor", "traidores", "traición",
    "ridículo", "ridícula", "patético", "patética", "lamentable",
    "nefasto", "nefasta", "desastre", "desastroso", "desastrosa",
    "escándalo", "escandaloso", "intolerable", "inaceptable",
    "indignante", "repugnante", "asqueroso", "asquerosa",
    "fascista", "fascismo", "tirano", "dictador", "dictadura",
    "extremista", "radical", "populista", "demagogo", "demagogia",
    "autoritario", "totalitario", "golpista", "golpe",
    "ultraderecha", "ultraizquierda", "sectario", "sectarismo",
    "ataque", "atacar", "destruir", "destrucción", "amenaza",
    "odio", "desprecio", "despreciar", "humillar", "humillación",
    "insulto", "insultar", "agredir", "agresión",
]

BLAME_WORDS = [
    "culpa", "culpable", "culpables", "responsable", "responsables",
    "responsabilidad", "rendir cuentas", "dimisión", "dimite", "dimita",
    "por culpa de", "es culpa de", "fracaso", "fracasado", "fracasada",
    "abandonar", "abandonado", "abandonados", "dejadez", "negligencia",
    "negligente", "incapaz", "incapaces", "permitir que", "consentir",
    "provocar", "provocado", "causado", "causar", "generar",
    "no ha hecho nada", "sin hacer nada", "inacción", "pasividad",
]


def count_keywords(text, keywords):
    text_lower = str(text).lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def has_keywords(text, keywords):
    return any(kw.lower() in str(text).lower() for kw in keywords)


def load_and_enrich(csv_path):
    df = pd.read_csv(csv_path)
    df["parsed_date"] = pd.to_datetime(
        df["date"], format="%a %b %d %H:%M:%S %z %Y", utc=True, errors="coerce")
    df["month"] = df["parsed_date"].dt.month
    df["year"] = df["parsed_date"].dt.year
    df = df[df["year"] == 2025].copy()
    df["aggression_count"] = df["text"].apply(lambda x: count_keywords(x, AGGRESSION_WORDS))
    df["blame_count"] = df["text"].apply(lambda x: count_keywords(x, BLAME_WORDS))
    df["has_aggression"] = df["aggression_count"] > 0
    df["has_blame"] = df["blame_count"] > 0
    df["hostility_score"] = df["aggression_count"] + df["blame_count"]
    df["is_crisis"] = df["month"].isin([6, 10, 11])
    return df


def main():
    # ══════════════════════════════════════════════════════════════════════
    # Load all datasets
    # ══════════════════════════════════════════════════════════════════════
    pol = load_and_enrich(os.path.join(OUTPUT_DIR, "all_politicians_2025.csv"))
    news = load_and_enrich(os.path.join(OUTPUT_DIR, "news_outlets_2025.csv"))

    pub = pd.read_csv(os.path.join(OUTPUT_DIR, "general_public_2025.csv"))
    pub["parsed_date"] = pd.to_datetime(
        pub["date"], format="%a %b %d %H:%M:%S %z %Y", utc=True, errors="coerce")
    pub["month"] = pub["parsed_date"].dt.month
    pub["aggression_count"] = pub["text"].apply(lambda x: count_keywords(x, AGGRESSION_WORDS))
    pub["blame_count"] = pub["text"].apply(lambda x: count_keywords(x, BLAME_WORDS))
    pub["has_aggression"] = pub["aggression_count"] > 0
    pub["has_blame"] = pub["blame_count"] > 0
    pub["hostility_score"] = pub["aggression_count"] + pub["blame_count"]
    crisis_periods = ["wildfires_june", "blackout_oct"]
    pub["is_crisis"] = pub["period"].isin(crisis_periods)

    MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    # ══════════════════════════════════════════════════════════════════════
    # SHEET 1: Figure 1 — Monthly Sentiment Trends
    # ══════════════════════════════════════════════════════════════════════
    months = list(range(1, 13))
    fig1 = pd.DataFrame({"Month_Num": months, "Month": [MONTH_NAMES[m] for m in months]})

    pol_sent = pol.groupby("month")["sentiment"].mean()
    news_sent = news.groupby("month")["sentiment"].mean()
    pub_sent = pub.groupby("month")["sentiment"].mean()

    fig1["Political_Actors"] = fig1["Month_Num"].map(pol_sent).round(3)
    fig1["News_Outlets"] = fig1["Month_Num"].map(news_sent).round(3)
    fig1["General_Public"] = fig1["Month_Num"].map(pub_sent).round(3)
    fig1["Crisis_Period"] = fig1["Month_Num"].map(
        lambda m: "Wildfires" if m == 6 else ("Power Outage" if m in [10, 11] else ""))

    # ══════════════════════════════════════════════════════════════════════
    # SHEET 2: Figure 1b — Monthly Aggression/Hostility Trends
    # ══════════════════════════════════════════════════════════════════════
    fig1b = pd.DataFrame({"Month_Num": months, "Month": [MONTH_NAMES[m] for m in months]})

    for label, df in [("Politicians", pol), ("News_Outlets", news), ("General_Public", pub)]:
        monthly = df.groupby("month").agg(
            aggression_pct=("has_aggression", lambda x: round(x.mean() * 100, 1)),
            blame_pct=("has_blame", lambda x: round(x.mean() * 100, 1)),
            hostility=("hostility_score", lambda x: round(x.mean(), 3)),
            n_tweets=("text", "count"),
        )
        fig1b[f"{label}_Aggression_%"] = fig1b["Month_Num"].map(monthly["aggression_pct"])
        fig1b[f"{label}_Blame_%"] = fig1b["Month_Num"].map(monthly["blame_pct"])
        fig1b[f"{label}_Hostility_Score"] = fig1b["Month_Num"].map(monthly["hostility"])
        fig1b[f"{label}_N_Tweets"] = fig1b["Month_Num"].map(monthly["n_tweets"])

    fig1b["Crisis_Period"] = fig1b["Month_Num"].map(
        lambda m: "Wildfires" if m == 6 else ("Power Outage" if m in [10, 11] else ""))

    # ══════════════════════════════════════════════════════════════════════
    # SHEET 3: Figure 2 — Before vs During Crisis (bar chart data)
    # ══════════════════════════════════════════════════════════════════════
    fig2_rows = []
    for label, df in [("Political Actors", pol), ("News Outlets", news), ("General Public", pub)]:
        before = df[~df["is_crisis"]]
        during = df[df["is_crisis"]]
        fig2_rows.append({
            "Group": label,
            "Before_Crisis_Sentiment": round(before["sentiment"].mean(), 3),
            "During_Crisis_Sentiment": round(during["sentiment"].mean(), 3),
            "Before_Crisis_Aggression_%": round(before["has_aggression"].mean() * 100, 1),
            "During_Crisis_Aggression_%": round(during["has_aggression"].mean() * 100, 1),
            "Before_Crisis_Blame_%": round(before["has_blame"].mean() * 100, 1),
            "During_Crisis_Blame_%": round(during["has_blame"].mean() * 100, 1),
            "Before_Crisis_Hostility": round(before["hostility_score"].mean(), 3),
            "During_Crisis_Hostility": round(during["hostility_score"].mean(), 3),
            "N_Before": len(before),
            "N_During": len(during),
        })
    fig2 = pd.DataFrame(fig2_rows)

    # ══════════════════════════════════════════════════════════════════════
    # SHEET 4: Figure 3 — President vs Opposition monthly
    # ══════════════════════════════════════════════════════════════════════
    president_names = ["Pedro Sanchez"]
    opp_pp = ["Alberto Nunez Feijoo", "Isabel Diaz Ayuso"]
    opp_vox = ["Santiago Abascal"]
    opp_left = ["Pablo Iglesias", "Irene Montero", "Ione Belarra"]
    all_opp = opp_pp + opp_vox + opp_left

    fig3 = pd.DataFrame({"Month_Num": months, "Month": [MONTH_NAMES[m] for m in months]})

    for col_label, names in [
        ("President", president_names),
        ("Opp_Avg", all_opp),
        ("Opp1_PP", opp_pp),
        ("Opp2_Vox", opp_vox),
        ("Opp3_Left", opp_left),
    ]:
        sub = pol[pol["politician"].isin(names)]
        # Sentiment
        sent = sub.groupby("month")["sentiment"].mean()
        fig3[f"{col_label}_Sentiment"] = fig3["Month_Num"].map(sent).round(3)
        # Aggression rate
        agg = sub.groupby("month")["has_aggression"].mean() * 100
        fig3[f"{col_label}_Aggression_%"] = fig3["Month_Num"].map(agg).round(1)
        # Hostility
        host = sub.groupby("month")["hostility_score"].mean()
        fig3[f"{col_label}_Hostility"] = fig3["Month_Num"].map(host).round(3)

    fig3["Crisis_Period"] = fig3["Month_Num"].map(
        lambda m: "Wildfires" if m == 6 else ("Power Outage" if m in [10, 11] else ""))

    # ══════════════════════════════════════════════════════════════════════
    # SHEET 5: President vs Opposition — Before/During summary
    # ══════════════════════════════════════════════════════════════════════
    fig3b_rows = []
    for label, names in [("President (Sanchez)", president_names), ("Opposition (All)", all_opp),
                         ("PP (Feijoo + Ayuso)", opp_pp), ("Vox (Abascal)", opp_vox),
                         ("Left (Iglesias, Montero, Belarra)", opp_left)]:
        sub = pol[pol["politician"].isin(names)]
        before = sub[~sub["is_crisis"]]
        during = sub[sub["is_crisis"]]
        fig3b_rows.append({
            "Actor": label,
            "Baseline_Aggression_%": round(before["has_aggression"].mean() * 100, 1) if len(before) else None,
            "Crisis_Aggression_%": round(during["has_aggression"].mean() * 100, 1) if len(during) else None,
            "Baseline_Blame_%": round(before["has_blame"].mean() * 100, 1) if len(before) else None,
            "Crisis_Blame_%": round(during["has_blame"].mean() * 100, 1) if len(during) else None,
            "Baseline_Hostility": round(before["hostility_score"].mean(), 3) if len(before) else None,
            "Crisis_Hostility": round(during["hostility_score"].mean(), 3) if len(during) else None,
            "N_Baseline": len(before),
            "N_Crisis": len(during),
        })
    fig3b = pd.DataFrame(fig3b_rows)

    # ══════════════════════════════════════════════════════════════════════
    # SHEET 6: Statistical Tests Summary
    # ══════════════════════════════════════════════════════════════════════
    from scipy import stats as sp_stats

    stat_rows = []
    for label, df in [("Political Actors", pol), ("News Outlets", news), ("General Public", pub)]:
        before = df[~df["is_crisis"]]
        during = df[df["is_crisis"]]

        # Chi-square for aggression
        ct_agg = pd.DataFrame({
            "agg": [during["has_aggression"].sum(), before["has_aggression"].sum()],
            "no":  [(~during["has_aggression"]).sum(), (~before["has_aggression"]).sum()],
        }, index=["crisis", "baseline"])
        chi2_a, p_a, _, _ = sp_stats.chi2_contingency(ct_agg)

        # Chi-square for blame
        ct_bl = pd.DataFrame({
            "bl": [during["has_blame"].sum(), before["has_blame"].sum()],
            "no": [(~during["has_blame"]).sum(), (~before["has_blame"]).sum()],
        }, index=["crisis", "baseline"])
        chi2_b, p_b, _, _ = sp_stats.chi2_contingency(ct_bl)

        # Mann-Whitney for hostility score
        u, p_u = sp_stats.mannwhitneyu(
            during["hostility_score"].dropna(), before["hostility_score"].dropna(), alternative="two-sided")

        stat_rows.append({
            "Group": label,
            "Test": "Aggression Rate",
            "Chi2": round(chi2_a, 2),
            "p_value": round(p_a, 4),
            "Significant (p<0.05)": "Yes" if p_a < 0.05 else "No",
        })
        stat_rows.append({
            "Group": label,
            "Test": "Blame Rate",
            "Chi2": round(chi2_b, 2),
            "p_value": round(p_b, 4),
            "Significant (p<0.05)": "Yes" if p_b < 0.05 else "No",
        })
        stat_rows.append({
            "Group": label,
            "Test": "Hostility Score (Mann-Whitney)",
            "Chi2": round(u, 2),
            "p_value": round(p_u, 4),
            "Significant (p<0.05)": "Yes" if p_u < 0.05 else "No",
        })
    fig_stats = pd.DataFrame(stat_rows)

    # ══════════════════════════════════════════════════════════════════════
    # SHEET 7: Summary / Overview
    # ══════════════════════════════════════════════════════════════════════
    summary_rows = [
        {"Metric": "Total Tweets Analyzed", "Value": len(pol) + len(news) + len(pub)},
        {"Metric": "Political Actor Tweets", "Value": len(pol)},
        {"Metric": "News Outlet Tweets", "Value": len(news)},
        {"Metric": "General Public Tweets", "Value": len(pub)},
        {"Metric": "Politicians Tracked", "Value": pol["politician"].nunique()},
        {"Metric": "News Outlets Tracked", "Value": news["outlet"].nunique() if "outlet" in news.columns else "N/A"},
        {"Metric": "Date Range", "Value": "Jan-Dec 2025"},
        {"Metric": "Crisis 1", "Value": "Wildfires — June 2025"},
        {"Metric": "Crisis 2", "Value": "Power Outage — October/November 2025"},
        {"Metric": "Sentiment Tool", "Value": "VADER (English baseline)"},
        {"Metric": "Aggression Detection", "Value": "Spanish keyword lexicon (57 terms)"},
        {"Metric": "Blame Detection", "Value": "Spanish keyword lexicon (32 terms)"},
    ]
    summary = pd.DataFrame(summary_rows)

    # ══════════════════════════════════════════════════════════════════════
    # WRITE TO EXCEL
    # ══════════════════════════════════════════════════════════════════════
    excel_path = os.path.join(OUTPUT_DIR, "poster_data_for_excel.xlsx")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Overview", index=False)
        fig1.to_excel(writer, sheet_name="Fig1_Monthly_Sentiment", index=False)
        fig1b.to_excel(writer, sheet_name="Fig1b_Monthly_Aggression", index=False)
        fig2.to_excel(writer, sheet_name="Fig2_Before_vs_During", index=False)
        fig3.to_excel(writer, sheet_name="Fig3_Pres_vs_Opp_Monthly", index=False)
        fig3b.to_excel(writer, sheet_name="Fig3b_Pres_vs_Opp_Summary", index=False)
        fig_stats.to_excel(writer, sheet_name="Statistical_Tests", index=False)

    print(f"Excel file saved -> {excel_path}")
    print(f"\nSheets:")
    print(f"  1. Overview — dataset summary")
    print(f"  2. Fig1_Monthly_Sentiment — line chart: 3 groups x 12 months")
    print(f"  3. Fig1b_Monthly_Aggression — line chart: aggression % + blame % per month")
    print(f"  4. Fig2_Before_vs_During — bar chart: before/during crisis comparison")
    print(f"  5. Fig3_Pres_vs_Opp_Monthly — line chart: President vs Opposition monthly")
    print(f"  6. Fig3b_Pres_vs_Opp_Summary — bar chart: President vs Opposition crisis summary")
    print(f"  7. Statistical_Tests — chi-square and Mann-Whitney results")

    # Also print Fig2 for quick reference
    print(f"\n--- Quick Reference: Figure 2 Data ---")
    print(fig2.to_string(index=False))
    print(f"\n--- Quick Reference: Figure 3b Data ---")
    print(fig3b.to_string(index=False))


if __name__ == "__main__":
    main()
