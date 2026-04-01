"""
analyze_politicians.py

Loads per-politician tweets_by_month.csv files and produces:
  Figure 1 — Monthly sentiment trend lines (one line per politician)
  Figure 2 — Average sentiment before vs during each crisis period (bar chart)
  Figure 3 — Category-level aggregate trends (President / Opposition / etc.)

Edit POLITICIANS and CRISIS_PERIODS below to match your data.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────────────────────────
YEAR = 2025

# Must match the handles / folders you scraped
POLITICIANS = [
    {"handle": "sanchezcastejon",  "label": "Pedro Sánchez",        "category": "President"},
    {"handle": "santi_abascal",    "label": "Santiago Abascal",     "category": "Opposition"},
    {"handle": "NunezFeijoo",      "label": "Alberto Núñez Feijóo", "category": "Opposition"},
    {"handle": "Yolanda_Diaz_",    "label": "Yolanda Díaz",         "category": "Government"},
]

# Month numbers (1-12) that count as crisis periods
CRISIS_PERIODS = {
    "Wildfires":    [5, 6],      # May–June
    "Power Outage": [10],        # October
}

OUTPUT_DIR = "figures"
# ──────────────────────────────────────────────────────────────────────────────

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Colour palette — extend if you add more politicians
COLORS = [
    "#1a6e3c",  # dark green
    "#c0392b",  # red
    "#2471a3",  # blue
    "#d4ac0d",  # gold
    "#7d3c98",  # purple
    "#ca6f1e",  # orange
    "#17a589",  # teal
    "#717d7e",  # grey
]


def load_all() -> dict:
    """Returns {handle: sentiment_summary DataFrame}. Skips missing folders."""
    data = {}
    for p in POLITICIANS:
        path = os.path.join(p["handle"], "sentiment_summary.csv")
        if not os.path.exists(path):
            print(f"[skip] {path} not found — run scrape_multi.py first.")
            continue
        df = pd.read_csv(path)
        df["handle"]   = p["handle"]
        df["label"]    = p["label"]
        df["category"] = p["category"]
        data[p["handle"]] = df
    return data


def full_year_series(df: pd.DataFrame) -> pd.Series:
    """Returns a 12-element Series of avg_sentiment indexed 1-12, NaN for missing months."""
    s = df.set_index("month")["avg_sentiment"]
    return s.reindex(range(1, 13))


def crisis_month_set() -> set:
    months = set()
    for v in CRISIS_PERIODS.values():
        months.update(v)
    return months


# ─── FIGURE 1: Individual monthly sentiment trends ────────────────────────────
def plot_figure1(data: dict):
    fig, ax = plt.subplots(figsize=(11, 5))

    crisis_months = crisis_month_set()

    # Shade crisis periods
    for name, months in CRISIS_PERIODS.items():
        ax.axvspan(min(months) - 0.5, max(months) + 0.5,
                   alpha=0.12, color="orange", zorder=0)
        ax.text((min(months) + max(months)) / 2, ax.get_ylim()[1] if ax.get_ylim()[1] != 1 else 0.32,
                name, ha="center", va="bottom", fontsize=8, color="darkorange", style="italic")

    for i, (handle, df) in enumerate(data.items()):
        series = full_year_series(df)
        label  = df["label"].iloc[0]
        color  = COLORS[i % len(COLORS)]
        ax.plot(range(1, 13), series.values, marker="o", markersize=4,
                label=label, color=color, linewidth=1.8)

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_ylabel("Average Sentiment Score (-1 to +1)")
    ax.set_xlabel(f"Month ({YEAR})")
    ax.set_title(f"Figure 1: Sentiment Trends — Individual Politicians, Spain {YEAR}")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.7)
    ax.set_ylim(-0.65, 0.45)
    ax.grid(axis="y", alpha=0.3)

    # Re-draw crisis labels now that ylim is set
    for name, months in CRISIS_PERIODS.items():
        ax.text((min(months) + max(months)) / 2, 0.38,
                name, ha="center", va="bottom", fontsize=8,
                color="darkorange", style="italic")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig1_individual_trends.png")
    fig.savefig(path, dpi=150)
    print(f"Saved → {path}")
    plt.close(fig)


# ─── FIGURE 2: Before vs During crisis — bar chart ────────────────────────────
def plot_figure2(data: dict):
    crisis_months = crisis_month_set()
    normal_months = set(range(1, 13)) - crisis_months

    handles = list(data.keys())
    labels  = [data[h]["label"].iloc[0] for h in handles]
    x       = np.arange(len(handles))
    width   = 0.35

    before_vals = []
    during_vals = []

    for handle in handles:
        df = data[handle]
        b  = df[df["month"].isin(normal_months)]["avg_sentiment"].mean()
        d  = df[df["month"].isin(crisis_months)]["avg_sentiment"].mean()
        before_vals.append(round(b, 3))
        during_vals.append(round(d, 3))

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_before = ax.bar(x - width/2, before_vals, width,
                         label="Before Crisis", color="#2e7d32", alpha=0.85)
    bars_during = ax.bar(x + width/2, during_vals, width,
                         label="During Crisis", color="#757575", alpha=0.85)

    # Value labels on bars
    for bar in bars_before:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_during:
        h = bar.get_height()
        offset = 0.005 if h >= 0 else -0.02
        ax.text(bar.get_x() + bar.get_width()/2, h + offset,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Average Sentiment Score (-1 to +1)")
    ax.set_title(f"Figure 2: Sentiment Before and During Crisis Periods, Spain {YEAR}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "fig2_before_vs_crisis.png")
    fig.savefig(path, dpi=150)
    print(f"Saved → {path}")
    plt.close(fig)


# ─── FIGURE 3: Category-level aggregate trends ────────────────────────────────
def plot_figure3(data: dict):
    # Build one avg_sentiment series per category
    cat_map = {}
    for p in POLITICIANS:
        h = p["handle"]
        if h not in data:
            continue
        cat = p["category"]
        cat_map.setdefault(cat, [])
        cat_map[cat].append(full_year_series(data[h]))

    if not cat_map:
        return

    fig, ax = plt.subplots(figsize=(11, 5))

    for name, months in CRISIS_PERIODS.items():
        ax.axvspan(min(months) - 0.5, max(months) + 0.5,
                   alpha=0.12, color="orange", zorder=0)
        ax.text((min(months) + max(months)) / 2, 0.38,
                name, ha="center", va="bottom", fontsize=8,
                color="darkorange", style="italic")

    for i, (category, series_list) in enumerate(cat_map.items()):
        combined = pd.concat(series_list, axis=1).mean(axis=1)
        ax.plot(range(1, 13), combined.values, marker="o", markersize=4,
                label=category, color=COLORS[i % len(COLORS)], linewidth=2)

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_ylabel("Average Sentiment Score (-1 to +1)")
    ax.set_xlabel(f"Month ({YEAR})")
    ax.set_title(f"Figure 3: Sentiment Trends by Actor Category, Spain {YEAR}")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.7)
    ax.set_ylim(-0.65, 0.45)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig3_category_trends.png")
    fig.savefig(path, dpi=150)
    print(f"Saved → {path}")
    plt.close(fig)


# ─── FIGURE 4: Per-crisis bar chart breakdown ─────────────────────────────────
def plot_figure4(data: dict):
    """One grouped bar chart per crisis period, showing each politician's avg sentiment."""
    handles = list(data.keys())
    labels  = [data[h]["label"].iloc[0] for h in handles]
    n_crises = len(CRISIS_PERIODS)

    fig, axes = plt.subplots(1, n_crises, figsize=(6 * n_crises, 5), sharey=True)
    if n_crises == 1:
        axes = [axes]

    for ax, (crisis_name, months) in zip(axes, CRISIS_PERIODS.items()):
        vals   = []
        colors = []
        for i, handle in enumerate(handles):
            df  = data[handle]
            avg = df[df["month"].isin(months)]["avg_sentiment"].mean()
            vals.append(round(avg, 3) if not pd.isna(avg) else 0)
            colors.append(COLORS[i % len(COLORS)])

        x    = np.arange(len(labels))
        bars = ax.bar(x, vals, color=colors, alpha=0.85, width=0.55)

        for bar in bars:
            h      = bar.get_height()
            offset = 0.006 if h >= 0 else -0.022
            ax.text(bar.get_x() + bar.get_width()/2, h + offset,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(f"{crisis_name} (months {months})")
        ax.set_ylabel("Avg Sentiment Score")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Figure 4: Sentiment During Each Crisis Period, Spain {YEAR}", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig4_per_crisis_breakdown.png")
    fig.savefig(path, dpi=150)
    print(f"Saved → {path}")
    plt.close(fig)


# ─── CONSOLE SUMMARY TABLE ────────────────────────────────────────────────────
def print_summary(data: dict):
    crisis_months = crisis_month_set()
    normal_months = set(range(1, 13)) - crisis_months

    print(f"\n{'═'*74}")
    print(f"  Sentiment Summary — @{YEAR}")
    print(f"{'═'*74}")
    print(f"{'Politician':<25} {'Category':<14} {'Overall':>9} {'Before':>9} {'During':>9}")
    print(f"{'─'*74}")

    for p in POLITICIANS:
        h = p["handle"]
        if h not in data:
            print(f"{p['label']:<25} {'(no data)'}")
            continue
        df      = data[h]
        overall = df["avg_sentiment"].mean()
        before  = df[df["month"].isin(normal_months)]["avg_sentiment"].mean()
        during  = df[df["month"].isin(crisis_months)]["avg_sentiment"].mean()
        print(f"{p['label']:<25} {p['category']:<14} {overall:>+9.3f} {before:>+9.3f} {during:>+9.3f}")

    print(f"{'═'*74}\n")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_all()
    if not data:
        print("No data loaded. Make sure you've run scrape_multi.py first.")
        return

    print(f"\nLoaded data for: {', '.join('@' + h for h in data)}")

    print_summary(data)
    plot_figure1(data)
    plot_figure2(data)
    plot_figure3(data)
    plot_figure4(data)

    print(f"\nAll figures saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
