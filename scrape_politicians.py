"""
Scrape tweets from top 10 Spanish politicians (2025).
Fixes twikit's broken ondemand.s regex for the new Twitter page format.
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
DATE_FROM = datetime(2025, 1, 1, tzinfo=timezone.utc)
DATE_TO   = datetime(2025, 12, 31, tzinfo=timezone.utc)
MAX_TWEETS_PER_POLITICIAN = 500

POLITICIANS = {
    "SanchezCastejon":  "Pedro Sanchez",
    "NunezFeijoo":      "Alberto Nunez Feijoo",
    "Santi_ABASCAL":    "Santiago Abascal",
    "Yolanda_Diaz_":    "Yolanda Diaz",
    "IdiazAyuso":       "Isabel Diaz Ayuso",
    "IreneMontero":     "Irene Montero",
    "PabloIglesias":    "Pablo Iglesias",
    "IoneBelarra":      "Ione Belarra",
    "AdaColau":         "Ada Colau",
    "gabrielrufian":    "Gabriel Rufian",
}
# ─────────────────────────────────────────────────────────────────────────────


# ─── MONKEYPATCH: fix twikit's broken ondemand.s detection ──────────────────
_original_get_indices = ClientTransaction.get_indices

INDICES_REGEX = re.compile(
    r"""(\(\w{1}\[(\d{1,2})\],\s*16\))+""", flags=(re.VERBOSE | re.MULTILINE))

async def _patched_get_indices(self, home_page_response, session, headers):
    """
    Twitter changed the page format: ondemand.s is now referenced as
    CHUNK_ID:"ondemand.s" in a name map and CHUNK_ID:"HASH" in a hash map,
    instead of the old "ondemand.s":"HASH" format.
    """
    try:
        return await _original_get_indices(self, home_page_response, session, headers)
    except Exception:
        pass

    page_text = str(home_page_response)

    # Step 1: find chunk ID for ondemand.s
    name_match = re.search(r'(\d+):"ondemand\.s"', page_text)
    if not name_match:
        raise Exception("Couldn't find ondemand.s chunk ID in page source")
    chunk_id = name_match.group(1)

    # Step 2: find the hash for that chunk ID
    hash_matches = re.findall(rf'[,{{]{chunk_id}:"([a-f0-9]{{6,12}})"', page_text)
    if not hash_matches:
        raise Exception(f"Couldn't find hash for chunk {chunk_id}")
    file_hash = hash_matches[0]

    # Step 3: fetch the JS file and extract indices
    on_demand_url = f"https://abs.twimg.com/responsive-web/client-web/ondemand.s.{file_hash}a.js"
    response = await session.request(method="GET", url=on_demand_url, headers=headers)
    key_byte_indices = []
    for item in INDICES_REGEX.finditer(response.text):
        key_byte_indices.append(item.group(2))

    if not key_byte_indices:
        raise Exception("Couldn't get KEY_BYTE indices from JS file")

    key_byte_indices = list(map(int, key_byte_indices))
    return key_byte_indices[0], key_byte_indices[1:]

ClientTransaction.get_indices = _patched_get_indices
# ─────────────────────────────────────────────────────────────────────────────


def convert_cookies(file):
    with open(file) as f:
        cookies = json.load(f)
    return {c["name"]: c["value"] for c in cookies}


def parse_tweet_date(date_str):
    return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")


def make_client():
    """Create a fresh twikit client with cookies."""
    client = Client(language="en")
    cookies = convert_cookies("cookies.json")
    client.set_cookies(cookies)
    return client


async def scrape_politician(client, username, display_name):
    """Scrape tweets for a single politician using search queries."""
    dataset = []

    quarters = [
        ("2025-01-01", "2025-04-01"),
        ("2025-04-01", "2025-07-01"),
        ("2025-07-01", "2025-10-01"),
        ("2025-10-01", "2026-01-01"),
    ]

    for since_str, until_str in quarters:
        query = f"from:{username} since:{since_str} until:{until_str}"
        print(f"  Querying: {query}")

        tweets = None
        for attempt in range(3):
            try:
                tweets = await client.search_tweet(query, "Latest")
                break
            except Exception as e:
                err = str(e)
                if "429" in err or "Rate limit" in err:
                    wait = 90 * (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s (attempt {attempt+1}/3)...")
                    await asyncio.sleep(wait)
                elif "recursion" in err:
                    print(f"  Recursion error, creating fresh client, waiting 45s...")
                    await asyncio.sleep(45)
                    client = make_client()
                elif "404" in err:
                    print(f"  404 error, creating fresh client, waiting 30s...")
                    await asyncio.sleep(30)
                    client = make_client()
                else:
                    print(f"  Error: {e}")
                    await asyncio.sleep(15)
                    break

        if not tweets:
            continue

        page_count = 0
        while tweets:
            for tweet in tweets:
                try:
                    tweet_date = parse_tweet_date(tweet.created_at)
                except Exception:
                    continue

                if not (DATE_FROM <= tweet_date <= DATE_TO):
                    continue

                hashtag_count = len(re.findall(r"#\w+", tweet.text))
                mention_count = len(re.findall(r"@\w+", tweet.text))
                link_count    = len(re.findall(r"http\S+", tweet.text))

                dataset.append({
                    "tweet_id":    tweet.id,
                    "username":    username,
                    "politician":  display_name,
                    "date":        tweet.created_at,
                    "text":        tweet.text,
                    "likes":       tweet.favorite_count,
                    "retweets":    tweet.retweet_count,
                    "replies":     tweet.reply_count,
                    "views":       getattr(tweet, "view_count", None),
                    "length":      len(tweet.text),
                    "hashtags":    hashtag_count,
                    "mentions":    mention_count,
                    "links":       link_count,
                })

            if len(dataset) >= MAX_TWEETS_PER_POLITICIAN:
                break

            page_count += 1
            if page_count >= 10:
                break

            try:
                tweets = await tweets.next()
            except Exception:
                break

            await asyncio.sleep(3)

        if len(dataset) >= MAX_TWEETS_PER_POLITICIAN:
            break

    return dataset, client


async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    client = make_client()
    analyzer = SentimentIntensityAnalyzer()
    all_data = []

    for username, display_name in POLITICIANS.items():
        # Skip if already scraped
        csv_path = os.path.join(OUTPUT_DIR, f"{username}.csv")
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path)
            if len(existing) > 0:
                print(f"\nSkipping @{username} — already have {len(existing)} tweets")
                all_data.append(existing)
                continue

        print(f"\n{'='*60}")
        print(f"Scraping: {display_name} (@{username})")
        print(f"{'='*60}")

        try:
            tweets, client = await scrape_politician(client, username, display_name)
        except Exception as e:
            print(f"  FAILED: {e}")
            client = make_client()
            continue

        if not tweets:
            print(f"  No tweets found for @{username}")
            continue

        df = pd.DataFrame(tweets)

        # Sentiment analysis
        df["sentiment"] = df["text"].apply(
            lambda x: analyzer.polarity_scores(x)["compound"]
        )
        df["sentiment_label"] = df["sentiment"].apply(
            lambda s: "positive" if s > 0.05 else ("negative" if s < -0.05 else "neutral")
        )

        # Save individual CSV
        df.to_csv(csv_path, index=False)
        print(f"  Saved {len(df)} tweets -> {csv_path}")
        print(f"  Date range: {df['date'].min()} -> {df['date'].max()}")
        print(f"  Sentiment: {df['sentiment_label'].value_counts().to_dict()}")

        all_data.append(df)

        # Longer delay between politicians to avoid rate limits
        print(f"  Waiting 90s before next politician...")
        await asyncio.sleep(90)

    # Combined dataset
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined_path = os.path.join(OUTPUT_DIR, "all_politicians_2025.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\n{'='*60}")
        print(f"COMBINED: {len(combined)} tweets -> {combined_path}")
        print(f"Tweets per politician:")
        print(combined.groupby("politician").size().to_string())
        print(f"{'='*60}")
    else:
        print("\nNo data collected for any politician.")


asyncio.run(main())
