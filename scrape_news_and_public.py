"""
Scrape tweets from Spanish news outlets and general public discourse
around the two 2025 crises (wildfires June, power outage October).
"""

import sys
sys.setrecursionlimit(5000)

from twikit import Client
from twikit.x_client_transaction.transaction import ClientTransaction
import asyncio
import json
import re
import os
import pandas as pd
from datetime import datetime, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── CONFIG ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = "spain_politicians_data"
MAX_TWEETS = 500

# Major Spanish news outlets — verified Twitter/X handles
NEWS_OUTLETS = {
    "el_pais":          "El Pais",
    "elmundoes":        "El Mundo",
    "abc_es":           "ABC",
    "LaVanguardia":     "La Vanguardia",
    "20m":              "20 Minutos",
    "EFEnoticias":      "Agencia EFE",
    "okdiario":         "OKDiario",
    "publico_es":       "Publico",
}

# General public search queries around crises and baseline periods
PUBLIC_QUERIES = {
    # Wildfires crisis (June 2025)
    "wildfires_june": [
        ("incendios España", "2025-06-01", "2025-07-01"),
        ("incendio forestal culpa", "2025-06-01", "2025-07-01"),
    ],
    # Power outage / blackout crisis (October 2025)
    "blackout_oct": [
        ("apagón España", "2025-10-15", "2025-11-15"),
        ("apagón culpa gobierno", "2025-10-15", "2025-11-15"),
        ("blackout Spain", "2025-10-15", "2025-11-15"),
    ],
    # Baseline non-crisis months for comparison
    "baseline_feb": [
        ("política España", "2025-02-01", "2025-03-01"),
        ("gobierno España", "2025-02-01", "2025-03-01"),
    ],
    "baseline_apr": [
        ("política España", "2025-04-01", "2025-05-01"),
        ("gobierno España", "2025-04-01", "2025-05-01"),
    ],
    "baseline_aug": [
        ("política España", "2025-08-01", "2025-09-01"),
        ("gobierno España", "2025-08-01", "2025-09-01"),
    ],
    "baseline_dec": [
        ("política España", "2025-12-01", "2026-01-01"),
        ("gobierno España", "2025-12-01", "2026-01-01"),
    ],
}
# ─────────────────────────────────────────────────────────────────────────────


# ─── MONKEYPATCH ─────────────────────────────────────────────────────────────
_original_get_indices = ClientTransaction.get_indices
INDICES_REGEX = re.compile(
    r"""(\(\w{1}\[(\d{1,2})\],\s*16\))+""", flags=(re.VERBOSE | re.MULTILINE))

async def _patched_get_indices(self, home_page_response, session, headers):
    try:
        return await _original_get_indices(self, home_page_response, session, headers)
    except Exception:
        pass
    page_text = str(home_page_response)
    name_match = re.search(r'(\d+):"ondemand\.s"', page_text)
    if not name_match:
        raise Exception("Couldn't find ondemand.s chunk ID")
    chunk_id = name_match.group(1)
    hash_matches = re.findall(rf'[,{{]{chunk_id}:"([a-f0-9]{{6,12}})"', page_text)
    if not hash_matches:
        raise Exception(f"Couldn't find hash for chunk {chunk_id}")
    url = f"https://abs.twimg.com/responsive-web/client-web/ondemand.s.{hash_matches[0]}a.js"
    response = await session.request(method="GET", url=url, headers=headers)
    indices = [item.group(2) for item in INDICES_REGEX.finditer(response.text)]
    if not indices:
        raise Exception("Couldn't get KEY_BYTE indices")
    indices = list(map(int, indices))
    return indices[0], indices[1:]

ClientTransaction.get_indices = _patched_get_indices
# ─────────────────────────────────────────────────────────────────────────────


def convert_cookies(file):
    with open(file) as f:
        cookies = json.load(f)
    return {c["name"]: c["value"] for c in cookies}


def make_client():
    client = Client(language="en")
    cookies = convert_cookies("cookies.json")
    client.set_cookies(cookies)
    return client


async def search_tweets(client, query, since, until, max_tweets=200):
    """Search tweets with retry logic. Returns (tweets_list, client)."""
    full_query = f"{query} since:{since} until:{until}"
    print(f"  Querying: {full_query}")

    dataset = []
    tweets = None

    for attempt in range(3):
        try:
            tweets = await client.search_tweet(full_query, "Latest")
            break
        except Exception as e:
            err = str(e)
            if "429" in err or "Rate limit" in err:
                wait = 90 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                await asyncio.sleep(wait)
            elif "recursion" in err or "404" in err:
                print(f"    Session error, recreating client, waiting 45s...")
                await asyncio.sleep(45)
                client = make_client()
            else:
                print(f"    Error: {e}")
                await asyncio.sleep(15)
                break

    if not tweets:
        return dataset, client

    page_count = 0
    while tweets:
        for tweet in tweets:
            try:
                dataset.append({
                    "tweet_id": tweet.id,
                    "date":     tweet.created_at,
                    "text":     tweet.text,
                    "likes":    tweet.favorite_count,
                    "retweets": tweet.retweet_count,
                    "replies":  tweet.reply_count,
                    "views":    getattr(tweet, "view_count", None),
                    "length":   len(tweet.text),
                    "hashtags": len(re.findall(r"#\w+", tweet.text)),
                    "mentions": len(re.findall(r"@\w+", tweet.text)),
                    "links":    len(re.findall(r"http\S+", tweet.text)),
                    "author":   getattr(tweet.user, "screen_name", "unknown") if hasattr(tweet, "user") and tweet.user else "unknown",
                })
            except Exception:
                continue

        if len(dataset) >= max_tweets:
            break

        page_count += 1
        if page_count >= 5:
            break

        try:
            tweets = await tweets.next()
        except Exception:
            break
        await asyncio.sleep(3)

    return dataset, client


async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = make_client()
    analyzer = SentimentIntensityAnalyzer()

    # ═══════════════════════════════════════════════════════
    # PART 1: News Outlets
    # ═══════════════════════════════════════════════════════
    news_csv = os.path.join(OUTPUT_DIR, "news_outlets_2025.csv")
    if os.path.exists(news_csv):
        print(f"News outlet data already exists, skipping...")
    else:
        print("\n" + "="*60)
        print("SCRAPING NEWS OUTLETS")
        print("="*60)

        all_news = []
        for handle, name in NEWS_OUTLETS.items():
            csv_path = os.path.join(OUTPUT_DIR, f"news_{handle}.csv")
            if os.path.exists(csv_path):
                print(f"\nSkipping @{handle} — already scraped")
                all_news.append(pd.read_csv(csv_path))
                continue

            print(f"\n--- {name} (@{handle}) ---")
            outlet_tweets = []

            # Scrape by monthly windows
            months = [
                ("2025-01-01", "2025-02-01"), ("2025-02-01", "2025-03-01"),
                ("2025-03-01", "2025-04-01"), ("2025-04-01", "2025-05-01"),
                ("2025-05-01", "2025-06-01"), ("2025-06-01", "2025-07-01"),
                ("2025-07-01", "2025-08-01"), ("2025-08-01", "2025-09-01"),
                ("2025-09-01", "2025-10-01"), ("2025-10-01", "2025-11-01"),
                ("2025-11-01", "2025-12-01"), ("2025-12-01", "2026-01-01"),
            ]
            for since, until in months:
                tweets, client = await search_tweets(
                    client, f"from:{handle}", since, until, max_tweets=60)
                for t in tweets:
                    t["outlet"] = name
                    t["handle"] = handle
                outlet_tweets.extend(tweets)

                if len(outlet_tweets) >= MAX_TWEETS:
                    break
                await asyncio.sleep(5)

            if outlet_tweets:
                df = pd.DataFrame(outlet_tweets)
                df["sentiment"] = df["text"].apply(
                    lambda x: analyzer.polarity_scores(x)["compound"])
                df["sentiment_label"] = df["sentiment"].apply(
                    lambda s: "positive" if s > 0.05 else ("negative" if s < -0.05 else "neutral"))
                df.to_csv(csv_path, index=False)
                print(f"  Saved {len(df)} tweets -> {csv_path}")
                all_news.append(df)
            else:
                print(f"  No tweets found for @{handle}")

            print(f"  Waiting 90s before next outlet...")
            await asyncio.sleep(90)

        if all_news:
            combined = pd.concat(all_news, ignore_index=True)
            combined.to_csv(news_csv, index=False)
            print(f"\nCombined news: {len(combined)} tweets -> {news_csv}")

    # ═══════════════════════════════════════════════════════
    # PART 2: General Public Tweets
    # ═══════════════════════════════════════════════════════
    public_csv = os.path.join(OUTPUT_DIR, "general_public_2025.csv")
    if os.path.exists(public_csv):
        print(f"\nPublic data already exists, skipping...")
    else:
        print("\n" + "="*60)
        print("SCRAPING GENERAL PUBLIC TWEETS")
        print("="*60)

        all_public = []
        for period_name, queries in PUBLIC_QUERIES.items():
            csv_path = os.path.join(OUTPUT_DIR, f"public_{period_name}.csv")
            if os.path.exists(csv_path):
                print(f"\nSkipping {period_name} — already scraped")
                all_public.append(pd.read_csv(csv_path))
                continue

            print(f"\n--- {period_name} ---")
            period_tweets = []
            for query_text, since, until in queries:
                tweets, client = await search_tweets(
                    client, query_text, since, until, max_tweets=200)
                for t in tweets:
                    t["period"] = period_name
                    t["query"] = query_text
                period_tweets.extend(tweets)
                await asyncio.sleep(60)

            if period_tweets:
                df = pd.DataFrame(period_tweets)
                df["sentiment"] = df["text"].apply(
                    lambda x: analyzer.polarity_scores(x)["compound"])
                df["sentiment_label"] = df["sentiment"].apply(
                    lambda s: "positive" if s > 0.05 else ("negative" if s < -0.05 else "neutral"))
                df.to_csv(csv_path, index=False)
                print(f"  Saved {len(df)} tweets -> {csv_path}")
                all_public.append(df)

            print(f"  Waiting 60s...")
            await asyncio.sleep(60)

        if all_public:
            combined = pd.concat(all_public, ignore_index=True)
            combined.to_csv(public_csv, index=False)
            print(f"\nCombined public: {len(combined)} tweets -> {public_csv}")

    print("\n" + "="*60)
    print("ALL SCRAPING COMPLETE")
    print("="*60)


asyncio.run(main())
