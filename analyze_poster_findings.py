"""
Deep analysis for the IFP Research Poster:
"To what extent do national crises in Spain intensify
 aggressive and blame-shifting discourse on Twitter (X)?"

Approach:
 1. Spanish-language aggression & blame keyword detection
 2. Engagement metrics as proxy for emotional reactivity
 3. Sentiment distribution shifts (% extreme negative)
 4. President vs Opposition divergence during crises
 5. Statistical significance tests
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import json
import re
from collections import Counter

OUTPUT_DIR = "spain_politicians_data"

# ═══════════════════════════════════════════════════════════════════════════════
# SPANISH AGGRESSION & BLAME LEXICONS
# ═══════════════════════════════════════════════════════════════════════════════

# Aggressive / hostile language keywords (Spanish)
AGGRESSION_WORDS = [
    # Direct aggression
    "vergüenza", "vergonzoso", "vergonzosa", "inútil", "inútiles",
    "incompetente", "incompetentes", "inepeto", "inepto", "ineptos",
    "mentira", "mentiras", "mentiroso", "mentirosa", "miente",
    "corrupto", "corrupta", "corrupción", "ladrón", "ladrones",
    "estafa", "estafador", "delincuente", "criminal", "criminales",
    "miserable", "sinvergüenza", "hipócrita", "hipócritas",
    "cobarde", "cobardes", "traidor", "traidores", "traición",
    "ridículo", "ridícula", "patético", "patética", "lamentable",
    "nefasto", "nefasta", "desastre", "desastroso", "desastrosa",
    "escándalo", "escandaloso", "intolerable", "inaceptable",
    "indignante", "repugnante", "asqueroso", "asquerosa",
    # Political aggression
    "fascista", "fascismo", "tirano", "dictador", "dictadura",
    "extremista", "radical", "populista", "demagogo", "demagogia",
    "autoritario", "totalitario", "golpista", "golpe",
    "ultraderecha", "ultraizquierda", "sectario", "sectarismo",
    # Conflict / confrontation
    "ataque", "atacar", "destruir", "destrucción", "amenaza",
    "odio", "desprecio", "despreciar", "humillar", "humillación",
    "insulto", "insultar", "agredir", "agresión",
]

# Blame-shifting keywords (Spanish)
BLAME_WORDS = [
    "culpa", "culpable", "culpables", "responsable", "responsables",
    "responsabilidad", "rendir cuentas", "dimisión", "dimite", "dimita",
    "por culpa de", "es culpa de", "fracaso", "fracasado", "fracasada",
    "abandonar", "abandonado", "abandonados", "dejadez", "negligencia",
    "negligente", "incapaz", "incapaces", "permitir que", "consentir",
    "provocar", "provocado", "causado", "causar", "generar",
    "no ha hecho nada", "sin hacer nada", "inacción", "pasividad",
]

# Crisis-specific keywords
CRISIS_KEYWORDS = {
    "wildfires": [
        "incendio", "incendios", "fuego", "forestal", "forestales",
        "quemar", "quemado", "arden", "llamas", "hectáreas",
        "bombero", "bomberos", "desalojo", "evacuación", "evacuados",
    ],
    "blackout": [
        "apagón", "apagon", "corte de luz", "sin luz", "electricidad",
        "energía", "eléctrico", "eléctrica", "suministro", "red eléctrica",
        "blackout", "oscuridad", "vela", "velas", "generador",
    ],
}


def count_keywords(text, keywords):
    """Count how many keywords appear in a text (case-insensitive)."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def has_keywords(text, keywords):
    """Check if any keyword appears in a text."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def load_data(csv_path):
    """Load and parse a CSV with date handling."""
    df = pd.read_csv(csv_path)
    df["parsed_date"] = pd.to_datetime(
        df["date"], format="%a %b %d %H:%M:%S %z %Y", utc=True, errors="coerce")
    df["month"] = df["parsed_date"].dt.month
    df["year"] = df["parsed_date"].dt.year
    df = df[df["year"] == 2025].copy()
    return df


def enrich_with_keywords(df):
    """Add aggression, blame, and crisis keyword columns."""
    df["aggression_count"] = df["text"].apply(lambda x: count_keywords(str(x), AGGRESSION_WORDS))
    df["blame_count"] = df["text"].apply(lambda x: count_keywords(str(x), BLAME_WORDS))
    df["has_aggression"] = df["aggression_count"] > 0
    df["has_blame"] = df["blame_count"] > 0
    df["hostility_score"] = df["aggression_count"] + df["blame_count"]

    # Crisis-specific
    df["mentions_wildfire"] = df["text"].apply(lambda x: has_keywords(str(x), CRISIS_KEYWORDS["wildfires"]))
    df["mentions_blackout"] = df["text"].apply(lambda x: has_keywords(str(x), CRISIS_KEYWORDS["blackout"]))

    # Crisis month flag
    df["is_crisis_month"] = df["month"].isin([6, 10, 11])
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_aggression_rates(df, group_name):
    """Compare aggression/blame rates during vs outside crisis months."""
    crisis = df[df["is_crisis_month"]]
    baseline = df[~df["is_crisis_month"]]

    if crisis.empty or baseline.empty:
        return None

    results = {
        "group": group_name,
        # Aggression
        "aggression_rate_baseline": round(baseline["has_aggression"].mean() * 100, 1),
        "aggression_rate_crisis": round(crisis["has_aggression"].mean() * 100, 1),
        "aggression_change_pct": round(
            (crisis["has_aggression"].mean() - baseline["has_aggression"].mean()) /
            max(baseline["has_aggression"].mean(), 0.001) * 100, 1),
        # Blame
        "blame_rate_baseline": round(baseline["has_blame"].mean() * 100, 1),
        "blame_rate_crisis": round(crisis["has_blame"].mean() * 100, 1),
        "blame_change_pct": round(
            (crisis["has_blame"].mean() - baseline["has_blame"].mean()) /
            max(baseline["has_blame"].mean(), 0.001) * 100, 1),
        # Hostility score
        "hostility_baseline": round(baseline["hostility_score"].mean(), 3),
        "hostility_crisis": round(crisis["hostility_score"].mean(), 3),
        # Sample sizes
        "n_baseline": len(baseline),
        "n_crisis": len(crisis),
    }

    # Chi-square test for aggression rate difference
    contingency = pd.DataFrame({
        "has_aggression": [crisis["has_aggression"].sum(), baseline["has_aggression"].sum()],
        "no_aggression": [(~crisis["has_aggression"]).sum(), (~baseline["has_aggression"]).sum()],
    }, index=["crisis", "baseline"])
    chi2, p_val, _, _ = stats.chi2_contingency(contingency)
    results["aggression_chi2"] = round(chi2, 2)
    results["aggression_p_value"] = round(p_val, 4)

    # Same for blame
    contingency_b = pd.DataFrame({
        "has_blame": [crisis["has_blame"].sum(), baseline["has_blame"].sum()],
        "no_blame": [(~crisis["has_blame"]).sum(), (~baseline["has_blame"]).sum()],
    }, index=["crisis", "baseline"])
    chi2_b, p_b, _, _ = stats.chi2_contingency(contingency_b)
    results["blame_chi2"] = round(chi2_b, 2)
    results["blame_p_value"] = round(p_b, 4)

    return results


def analyze_engagement_reactivity(df, group_name):
    """Compare engagement metrics during vs outside crisis months."""
    crisis = df[df["is_crisis_month"]]
    baseline = df[~df["is_crisis_month"]]

    if crisis.empty or baseline.empty:
        return None

    metrics = {}
    for col in ["likes", "retweets", "replies"]:
        b_med = baseline[col].median()
        c_med = crisis[col].median()
        metrics[f"{col}_baseline_median"] = b_med
        metrics[f"{col}_crisis_median"] = c_med
        metrics[f"{col}_change_pct"] = round(
            (c_med - b_med) / max(b_med, 1) * 100, 1)

        # Mann-Whitney U test
        try:
            u, p = stats.mannwhitneyu(crisis[col].dropna(), baseline[col].dropna(), alternative="two-sided")
            metrics[f"{col}_p_value"] = round(p, 4)
        except Exception:
            metrics[f"{col}_p_value"] = None

    metrics["group"] = group_name
    return metrics


def analyze_president_vs_opposition(df):
    """Detailed president vs opposition analysis during crises."""
    president_names = ["Pedro Sanchez"]
    opposition_names = [
        "Alberto Nunez Feijoo", "Isabel Diaz Ayuso",
        "Santiago Abascal",
        "Pablo Iglesias", "Irene Montero", "Ione Belarra",
    ]

    pres = df[df["politician"].isin(president_names)]
    opp = df[df["politician"].isin(opposition_names)]

    results = {}

    for period, label in [(True, "crisis"), (False, "baseline")]:
        p = pres[pres["is_crisis_month"] == period]
        o = opp[opp["is_crisis_month"] == period]

        results[f"president_{label}_aggression"] = round(p["has_aggression"].mean() * 100, 1) if len(p) else None
        results[f"president_{label}_blame"] = round(p["has_blame"].mean() * 100, 1) if len(p) else None
        results[f"president_{label}_hostility"] = round(p["hostility_score"].mean(), 3) if len(p) else None
        results[f"president_{label}_n"] = len(p)

        results[f"opposition_{label}_aggression"] = round(o["has_aggression"].mean() * 100, 1) if len(o) else None
        results[f"opposition_{label}_blame"] = round(o["has_blame"].mean() * 100, 1) if len(o) else None
        results[f"opposition_{label}_hostility"] = round(o["hostility_score"].mean(), 3) if len(o) else None
        results[f"opposition_{label}_n"] = len(o)

    return results


def analyze_monthly_hostility(df, group_name):
    """Monthly breakdown of hostility metrics."""
    monthly = df.groupby("month").agg(
        aggression_rate=("has_aggression", "mean"),
        blame_rate=("has_blame", "mean"),
        avg_hostility=("hostility_score", "mean"),
        n_tweets=("text", "count"),
    ).round(3)
    monthly["aggression_rate"] = (monthly["aggression_rate"] * 100).round(1)
    monthly["blame_rate"] = (monthly["blame_rate"] * 100).round(1)
    monthly["group"] = group_name
    return monthly


def find_top_aggressive_words(df, label):
    """Find most common aggression/blame words used."""
    all_agg = Counter()
    all_blame = Counter()
    for text in df["text"].dropna():
        text_lower = text.lower()
        for w in AGGRESSION_WORDS:
            if w.lower() in text_lower:
                all_agg[w] += 1
        for w in BLAME_WORDS:
            if w.lower() in text_lower:
                all_blame[w] += 1

    print(f"\n  Top aggression words ({label}):")
    for word, count in all_agg.most_common(10):
        print(f"    {word}: {count}")
    print(f"  Top blame words ({label}):")
    for word, count in all_blame.most_common(10):
        print(f"    {word}: {count}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("DEEP ANALYSIS: Aggression & Blame-Shifting in Spanish Crises 2025")
    print("=" * 70)

    all_aggression_results = []
    all_engagement_results = []
    all_monthly = []

    # ── Load & enrich all datasets ──
    datasets = {}

    pol_path = os.path.join(OUTPUT_DIR, "all_politicians_2025.csv")
    if os.path.exists(pol_path):
        datasets["Political Actors"] = enrich_with_keywords(load_data(pol_path))

    news_path = os.path.join(OUTPUT_DIR, "news_outlets_2025.csv")
    if os.path.exists(news_path):
        datasets["News Outlets"] = enrich_with_keywords(load_data(news_path))

    pub_path = os.path.join(OUTPUT_DIR, "general_public_2025.csv")
    if os.path.exists(pub_path):
        pub = pd.read_csv(pub_path)
        pub["parsed_date"] = pd.to_datetime(
            pub["date"], format="%a %b %d %H:%M:%S %z %Y", utc=True, errors="coerce")
        pub["month"] = pub["parsed_date"].dt.month
        pub["year"] = pub["parsed_date"].dt.year
        # Map period to crisis/baseline
        crisis_periods = ["wildfires_june", "blackout_oct"]
        pub["is_crisis_period"] = pub["period"].isin(crisis_periods)
        pub = enrich_with_keywords(pub)
        # Override is_crisis_month with period-based flag
        pub["is_crisis_month"] = pub["is_crisis_period"]
        datasets["General Public"] = pub

    # ══════════════════════════════════════════════════════════════════════
    # FINDING 1: Aggression & Blame rates during crises
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINDING 1: Aggression & Blame Rates — Crisis vs Baseline")
    print("=" * 70)

    for name, df in datasets.items():
        result = analyze_aggression_rates(df, name)
        if result:
            all_aggression_results.append(result)
            print(f"\n  {name} (baseline n={result['n_baseline']}, crisis n={result['n_crisis']}):")
            print(f"    Aggression rate: {result['aggression_rate_baseline']}% → {result['aggression_rate_crisis']}% "
                  f"({result['aggression_change_pct']:+.1f}% change, p={result['aggression_p_value']})")
            print(f"    Blame rate:      {result['blame_rate_baseline']}% → {result['blame_rate_crisis']}% "
                  f"({result['blame_change_pct']:+.1f}% change, p={result['blame_p_value']})")
            print(f"    Hostility score: {result['hostility_baseline']} → {result['hostility_crisis']}")

    agg_df = pd.DataFrame(all_aggression_results)
    agg_df.to_csv(os.path.join(OUTPUT_DIR, "finding1_aggression_rates.csv"), index=False)

    # ══════════════════════════════════════════════════════════════════════
    # FINDING 2: Engagement / Emotional Reactivity
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINDING 2: Engagement Reactivity — Crisis vs Baseline")
    print("=" * 70)

    for name, df in datasets.items():
        result = analyze_engagement_reactivity(df, name)
        if result:
            all_engagement_results.append(result)
            print(f"\n  {name}:")
            for metric in ["likes", "retweets", "replies"]:
                print(f"    {metric}: {result[f'{metric}_baseline_median']} → {result[f'{metric}_crisis_median']} "
                      f"({result[f'{metric}_change_pct']:+.1f}%, p={result[f'{metric}_p_value']})")

    eng_df = pd.DataFrame(all_engagement_results)
    eng_df.to_csv(os.path.join(OUTPUT_DIR, "finding2_engagement.csv"), index=False)

    # ══════════════════════════════════════════════════════════════════════
    # FINDING 3: President vs Opposition during crises
    # ══════════════════════════════════════════════════════════════════════
    if "Political Actors" in datasets:
        print("\n" + "=" * 70)
        print("FINDING 3: President vs Opposition — Crisis Behavior")
        print("=" * 70)

        pol_df = datasets["Political Actors"]
        pvo = analyze_president_vs_opposition(pol_df)

        print(f"\n  PRESIDENT (Sanchez):")
        print(f"    Baseline: aggression={pvo['president_baseline_aggression']}%, "
              f"blame={pvo['president_baseline_blame']}%, "
              f"hostility={pvo['president_baseline_hostility']} (n={pvo['president_baseline_n']})")
        print(f"    Crisis:   aggression={pvo['president_crisis_aggression']}%, "
              f"blame={pvo['president_crisis_blame']}%, "
              f"hostility={pvo['president_crisis_hostility']} (n={pvo['president_crisis_n']})")

        print(f"\n  OPPOSITION (Feijoo, Ayuso, Abascal, Iglesias, Montero, Belarra):")
        print(f"    Baseline: aggression={pvo['opposition_baseline_aggression']}%, "
              f"blame={pvo['opposition_baseline_blame']}%, "
              f"hostility={pvo['opposition_baseline_hostility']} (n={pvo['opposition_baseline_n']})")
        print(f"    Crisis:   aggression={pvo['opposition_crisis_aggression']}%, "
              f"blame={pvo['opposition_crisis_blame']}%, "
              f"hostility={pvo['opposition_crisis_hostility']} (n={pvo['opposition_crisis_n']})")

        pvo_df = pd.DataFrame([pvo])
        pvo_df.to_csv(os.path.join(OUTPUT_DIR, "finding3_president_vs_opposition.csv"), index=False)

        # Monthly breakdown for Figure 3
        for actor_type, names in [
            ("President", ["Pedro Sanchez"]),
            ("Opposition", ["Alberto Nunez Feijoo", "Isabel Diaz Ayuso", "Santiago Abascal",
                            "Pablo Iglesias", "Irene Montero", "Ione Belarra"]),
        ]:
            sub = pol_df[pol_df["politician"].isin(names)]
            monthly = sub.groupby("month").agg(
                aggression_rate=("has_aggression", lambda x: round(x.mean() * 100, 1)),
                blame_rate=("has_blame", lambda x: round(x.mean() * 100, 1)),
                avg_hostility=("hostility_score", lambda x: round(x.mean(), 3)),
                n=("text", "count"),
            )
            print(f"\n  {actor_type} Monthly Aggression Rate (%):")
            print(f"  {monthly[['aggression_rate','blame_rate','avg_hostility','n']].to_string()}")

    # ══════════════════════════════════════════════════════════════════════
    # FINDING 4: Monthly hostility trends (for Figure 1 replacement)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINDING 4: Monthly Hostility Trends (All Groups)")
    print("=" * 70)

    monthly_all = {}
    for name, df in datasets.items():
        monthly = analyze_monthly_hostility(df, name)
        monthly_all[name] = monthly
        print(f"\n  {name}:")
        print(f"  {monthly.to_string()}")

    # Build combined monthly table
    months = list(range(1, 13))
    fig1_new = pd.DataFrame({"month": months})
    for name, monthly in monthly_all.items():
        fig1_new[f"{name}_aggression_pct"] = fig1_new["month"].map(monthly["aggression_rate"])
        fig1_new[f"{name}_blame_pct"] = fig1_new["month"].map(monthly["blame_rate"])
        fig1_new[f"{name}_hostility"] = fig1_new["month"].map(monthly["avg_hostility"])
    fig1_new.to_csv(os.path.join(OUTPUT_DIR, "finding4_monthly_hostility.csv"), index=False)
    print(f"\n  Saved -> finding4_monthly_hostility.csv")

    # ══════════════════════════════════════════════════════════════════════
    # FINDING 5: Top words used
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINDING 5: Most Common Aggression & Blame Words")
    print("=" * 70)

    for name, df in datasets.items():
        crisis = df[df["is_crisis_month"]]
        baseline = df[~df["is_crisis_month"]]
        find_top_aggressive_words(crisis, f"{name} — CRISIS")
        find_top_aggressive_words(baseline, f"{name} — BASELINE")

    # ══════════════════════════════════════════════════════════════════════
    # POSTER-READY SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("POSTER-READY DATA SUMMARY")
    print("=" * 70)

    total = sum(len(df) for df in datasets.values())
    print(f"\n  Total tweets analyzed: {total}")
    print(f"  Groups: {', '.join(datasets.keys())}")
    print(f"  Period: January - December 2025")
    print(f"  Crises: Wildfires (June), Power Outage (Oct-Nov)")
    print(f"  Method: Spanish-language keyword-based aggression/blame detection")
    print(f"          + VADER sentiment + engagement analysis")
    print(f"          + chi-square / Mann-Whitney statistical tests")


if __name__ == "__main__":
    main()
